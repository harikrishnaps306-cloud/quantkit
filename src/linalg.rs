//! Small numerical helpers shared by the Monte Carlo and surface modules.

/// Solves the 3x3 system `a x = b` by Gaussian elimination with partial
/// pivoting. `None` if the matrix is (numerically) singular.
pub(crate) fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (x, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let s: f64 = (row + 1..3).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// Nelder-Mead minimisation of `f` starting from `x0` with initial step
/// `step`. Returns the best point and its value.
pub(crate) fn nelder_mead<const N: usize, F: Fn(&[f64; N]) -> f64>(
    f: F,
    x0: [f64; N],
    step: f64,
    max_iter: usize,
) -> ([f64; N], f64) {
    let mut simplex: Vec<([f64; N], f64)> = Vec::with_capacity(N + 1);
    simplex.push((x0, f(&x0)));
    for i in 0..N {
        let mut x = x0;
        x[i] += step;
        simplex.push((x, f(&x)));
    }
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (best, worst) = (simplex[0].1, simplex[N].1);
        if (worst - best).abs() <= 1e-18 + 1e-14 * best.abs() {
            break;
        }
        let mut centroid = [0.0; N];
        for (x, _) in &simplex[..N] {
            for j in 0..N {
                centroid[j] += x[j] / N as f64;
            }
        }
        let point = |t: f64| -> [f64; N] {
            let mut p = [0.0; N];
            for j in 0..N {
                p[j] = centroid[j] + t * (simplex[N].0[j] - centroid[j]);
            }
            p
        };
        let xr = point(-1.0);
        let fr = f(&xr);
        if fr < simplex[0].1 {
            let xe = point(-2.0);
            let fe = f(&xe);
            simplex[N] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[N - 1].1 {
            simplex[N] = (xr, fr);
        } else {
            let (xc, fc) = if fr < simplex[N].1 {
                let x = point(-0.5);
                (x, f(&x))
            } else {
                let x = point(0.5);
                (x, f(&x))
            };
            if fc < simplex[N].1.min(fr) {
                simplex[N] = (xc, fc);
            } else {
                let b = simplex[0].0;
                for item in simplex.iter_mut().skip(1) {
                    for (x, bj) in item.0.iter_mut().zip(&b) {
                        *x = bj + 0.5 * (*x - bj);
                    }
                    item.1 = f(&item.0);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve3_solves_and_detects_singular() {
        let x = solve3(
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            [2.0, 6.0, 12.0],
        )
        .unwrap();
        assert_eq!(x, [1.0, 2.0, 3.0]);
        assert!(solve3([[1.0, 1.0, 1.0]; 3], [1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn nelder_mead_finds_quadratic_minimum() {
        let (x, fx) = nelder_mead(
            |p: &[f64; 2]| (p[0] - 1.5).powi(2) + 3.0 * (p[1] + 0.5).powi(2),
            [0.0, 0.0],
            0.5,
            500,
        );
        assert!((x[0] - 1.5).abs() < 1e-6 && (x[1] + 0.5).abs() < 1e-6);
        assert!(fx < 1e-11);
    }
}
