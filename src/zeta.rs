// SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

use core::{clone::Clone, f64::consts};

use libm::{fmax, ldexp, remainder};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, OperationNorm};
use num_complex::{c64, Complex64};
use num_traits::identities;
use special::Gamma;

use crate::crandall;

/// calculates the first sum in Crandall's formula.
///
/// * @brief calculates the first sum in Crandall's formula.
/// * @param[in] nu: exponent for the Epstein zeta function.
/// * @param[in] dim: dimension of the input vectors.
/// * @param[in] lambda: parameters that decides the weight of each sum.
/// * @param[in] m: matrix that transforms the lattice in the Epstein Zeta
/// * function.
/// * @param[in] x: projection of x vector to elementary lattice cell.
/// * @param[in] y: projection of y vector to elementary lattice cell.
/// * @param[in] cutoffs: how many summands in each direction are considered.
/// * @param[in] zArgBound: global bound on when to use the asymptotic expansion in
/// * the incomplete gamma evaluation.
/// * @return helper function for the first sum in crandalls formula. Calculates
/// * sum_{z in m whole_numbers ** dim} G_{nu}((z - x) / lambda))
/// * X exp(-2 * PI * I * z * y)
fn sum_real(
    nu: f64,
    lambda: f64,
    m: &Array2<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    cutoffs: &Array1<isize>,
    z_arg_bound: f64,
) -> Complex64 {
    let dim = m.ncols();
    let mut zv = Array1::default(dim);
    // cuboid cutoffs
    let mut total_summands = 1;
    let mut total_cutoffs = Array1::zeros(dim);
    for k in 0..dim {
        total_cutoffs[k] = total_summands;
        total_summands *= 2 * cutoffs[k] + 1;
    }
    let mut sum = c64(0.0, 0.0);
    let mut epsilon = c64(0.0, 0.0);
    // First Sum (in real space)
    for n in 0..total_summands {
        for k in 0..dim {
            zv[k] = ((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k];
        }
        // println!("zv {:?}", zv);
        let zvf = zv.mapv(|elem| elem as f64);
        let mut lv = m.dot(&zvf);
        let rot = c64(0.0, -2.0 * consts::PI * lv.dot(y)).exp();
        lv -= x;
        // summing using Kahan's method
        let auxy = rot * crandall::crandall_g(nu, &lv, 1.0 / lambda, z_arg_bound) - epsilon;
        let auxt = sum + auxy;
        epsilon = (auxt - sum) - auxy;
        sum = auxt;
        // println!(
        //     "{} rot {:?} auxy {:?} auxt {:?} epsilon {:?}",
        //     n, rot, auxy, auxt, epsilon
        // );
    }
    sum
}

/// calculates the second sum in Crandall's formula.
///
/// * @param[in] nu: exponent for the Epstein zeta function.
/// * @param[in] dim: dimension of the input vectors.
/// * @param[in] lambda: parameters that decides the weight of each sum.
/// * @param[in] m: matrix that transforms the lattice in the Epstein Zeta
/// * function.
/// * @param[in] x: projection of x vector to elementary lattice cell.
/// * @param[in] y: projection of y vector to elementary lattice cell.
/// * @param[in] cutoffs: how many summands in each direction are considered.
/// * @param[in] zArgBound: global bound on when to use the asymptotic expansion in
/// * the incomplete gamma evaluation.
/// * @return helper function for the second sum in crandalls formula. Calculates
/// * sum_{k in m_invt whole_numbers ** dim without zero} G_{dim - nu}(lambda * (k + y))
/// * X exp(-2 * PI * I * x * (k + y))
fn sum_fourier(
    nu: f64,
    lambda: f64,
    m_invt: &Array2<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    cutoffs: &Array1<isize>,
    z_arg_bound: f64,
) -> Complex64 {
    let dim = y.len();
    let mut zv = Array1::default(dim); // counting vector in Z^dim
                                       // let lv = Array1::default(dim); // lattice vector
    let mut total_summands = 1;
    let mut total_cutoffs = Vec::new();
    for k in 0..dim {
        total_cutoffs.push(total_summands);
        total_summands *= 2 * cutoffs[k] + 1;
    }
    let zero_index = (total_summands - 1) / 2;
    let mut sum = c64(0.0, 0.0);
    let mut epsilon = c64(0.0, 0.0);
    // second sum (in fourier space)
    for n in 0..zero_index {
        for k in 0..dim {
            zv[k] = ((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k];
        }
        let zvf = zv.mapv(|elem| elem as f64);
        let lv = m_invt.dot(&zvf) + y;
        let rot = c64(0.0, -2.0 * consts::PI * lv.dot(x));
        let auxy = rot * crandall::crandall_g(dim as f64 - nu, &lv, lambda, z_arg_bound) - epsilon;
        let auxt = sum + auxy;
        epsilon = (auxt - sum) - auxy;
        sum = auxt;
    }
    // skips zero
    // TODO this is all just the above; make a function of n
    for n in zero_index + 1..total_summands {
        for k in 0..dim {
            zv[k] = ((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k];
        }
        let zvf = zv.mapv(|elem| elem as f64);
        let lv = m_invt.dot(&zvf) + y;
        let rot = c64(0.0, -2.0 * consts::PI * lv.dot(x));
        let auxy = rot * crandall::crandall_g(dim as f64 - nu, &lv, lambda, z_arg_bound) - epsilon;
        let auxt = sum + auxy;
        epsilon = (auxt - sum) - auxy;
        sum = auxt;
    }
    sum
}

/// calculate projection of vector to elementary lattice cell.
///
/// * @param[in] dim: dimension of the input vectors
/// * @param[in] m: matrix that transforms the lattice in the function.
/// * @param[in] m_invt: inverse of m.
/// * @param[in] v: vector for which the projection to the elementary lattice cell
/// * is needed.
/// * @return projection of v to the elementary lattice cell.
fn vector_proj(m: &Array2<f64>, m_invt: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let dim = v.len();
    let mut todo = false;
    let mut vt = Array1::<f64>::zeros(dim);
    for i in 0..dim {
        for j in 0..dim {
            vt[i] += m_invt[(j, i)] * v[j];
        }
    }
    // check if projection is needed, else return
    let mut i = 0;
    while i < dim && !todo {
        todo = todo || (vt[i] <= -0.5 || vt[i] >= 0.5);
        i += 1;
    }
    if todo {
        for i in 0..dim {
            vt[i] = remainder(vt[i], 1.0);
        }
        let mut vres = Array1::zeros(dim);
        for i in 0..dim {
            for j in 0..dim {
                vres[i] += m[(i, j)] * vt[j]
            }
        }
        vres
    } else {
        v.clone()
    }
}

fn is_diagonal<T: PartialEq + identities::Zero>(m: &Array2<T>) -> bool {
    let mut ret = true;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            ret = ret && ((i == j) || (m[(i, j)] == T::zero()));
        }
    }
    ret
}

/// Smallest value z such that G(nu, z) is negligible for nu < 10.
const G_BOUND: f64 = 3.2;

/// calculates the (regularized) Epstein Zeta function.
///
/// * @param[in] nu: exponent for the Epstein zeta function.
/// * @param[in] dim: dimension of the input vectors.
/// * @param[in] m: matrix that transforms the lattice in the Epstein Zeta
/// * function.
/// * @param[in] x: x vector of the Epstein Zeta function.
/// * @param[in] y: y vector of the Epstein Zeta function.
/// * @param[in] lambda: relative weight of the sums in Crandall's formula.
/// * @param[in] reg: 0 for no regularization, > 0 for the regularization.
/// * @return function value of the regularized Epstein zeta.
fn epstein_zeta_internal(
    nu: f64,
    m: &Array2<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    lambda: f64,
    reg: bool,
) -> Complex64 {
    let dim = x.len();
    // 1. Transform: Compute determinant and fourier transformed matrix, scale
    // both of them
    let mut m_real = m.clone();
    let is_diagonal = is_diagonal(m);
    let mut m_fourier = m.inv().unwrap();
    let vol = m.diag().product().abs();
    let ms = vol.powi(-1 / dim as i32);
    m_real *= ms;
    m_fourier *= ms;
    let x_t1 = x * ms;
    let y_t1 = y / ms;
    // 2. transform: get x and y in their respective elementary cells
    let x_t2 = vector_proj(&m_real, &m_fourier, &x_t1);
    let y_t2 = vector_proj(&m_fourier, &m_real, &y_t1);
    // set cutoffs
    let cutoff_id = G_BOUND + 0.5;
    // If diagonal, choose absolute diag. entries for cutoff
    // Else, choose cutoff depending on smallest and biggest abs eigenvalue
    let cutoffs_real = if is_diagonal {
        (cutoff_id / m_real.diag().abs()).floor()
    } else {
        let ev_abs_min_r = m_real.opnorm_inf().unwrap();
        Array1::ones(m_real.diag().len()) * cutoff_id * ev_abs_min_r
    };
    let cutoffs_fourier = if is_diagonal {
        (cutoff_id * m_real.diag().abs()).floor()
    } else {
        let ev_abs_max = m_fourier.opnorm_inf().unwrap();
        Array1::ones(m_real.diag().len()) * cutoff_id * ev_abs_max
    };
    let cutoffs_u_real = cutoffs_real.mapv(|elem| elem as isize);
    let cutoffs_u_fourier = cutoffs_fourier.mapv(|elem| elem as isize);
    // handle special case of non-positive integer values nu.
    let mut res = c64(0.0, 0.0);
    /// epsilon for the cutoff around nu = dimension.
    let EPS = ldexp(1.0, -30);
    /// epsilon for the cutoff around x = 0 and y = 0
    let EPS_ZERO_Y = 1.0e-64;
    let dimf = dim as f64;
    if nu < 1.0 && ((nu / 2.0) - (nu / 2.0).round_ties_even()).abs() < EPS {
        res = -c64(0.0, -2.0 * consts::PI * x_t1.dot(&y_t2)).exp();
    } else if (nu - dimf).abs() < EPS && y_t2.dot(&y_t2) < EPS_ZERO_Y && !reg {
        res = c64(0.0, 0.0);
    } else {
        let z_arg_bound = crandall::assign_z_arg_bound(nu);
        let vx = x_t1.clone() - x_t2.clone();
        let mut xfactor = c64(0.0, -2.0 * consts::PI * vx.dot(&y_t1)).exp();
        let s1: Complex64;
        let mut s2: Complex64;
        if reg {
            let nc = crandall::crandall_g_reg(dimf - nu, &y_t1, lambda);
            let rot = c64(0.0, 2.0 * consts::PI * x_t1.dot(&y_t1));
            s2 = sum_fourier(
                nu,
                lambda,
                &m_fourier,
                &x_t1,
                &y_t2,
                &cutoffs_u_fourier,
                z_arg_bound,
            );
            // correct wrong zero summand in regularized fourier sum.
            if y_t1 != y_t2 {
                s2 += crandall::crandall_g(dimf - nu, &y_t2, lambda, z_arg_bound)
                    * c64(0.0, -2.0 * consts::PI * x_t1.dot(&y_t2))
                    - crandall::crandall_g(dimf - nu, &y_t1, lambda, z_arg_bound)
                        * c64(0.0, -2.0 * consts::PI * x_t1.dot(&y_t1));
            }
            s2 *= rot + nc;
            s1 = sum_real(
                nu,
                lambda,
                &m_real,
                &x_t2,
                &y_t2,
                &cutoffs_u_real,
                z_arg_bound,
            ) * rot
                * xfactor;
            xfactor = c64(1.0, 1.0);
        } else {
            let nc = crandall::crandall_g(dimf - nu, &y_t2, lambda, z_arg_bound);
            s1 = sum_real(
                nu,
                lambda,
                &m_real,
                &x_t2,
                &y_t2,
                &cutoffs_u_real,
                z_arg_bound,
            );
            s2 = sum_fourier(
                nu,
                lambda,
                &m_fourier,
                &x_t2,
                &y_t2,
                &cutoffs_u_fourier,
                z_arg_bound,
            ) + nc;
        }
        res = xfactor * (lambda * lambda / consts::PI).powf(-nu / 2.0) / Gamma::gamma(nu / 2.0)
            * (s1 + lambda.powf(dimf) * s2);
    }
    res *= ms.powf(nu);
    // apply correction to matrix scaling if nu = d + 2k
    let k = fmax(0.0, ((nu - dimf) / 2.0).round_ties_even());
    if reg && (nu == (dimf + 2.0 * k)) {
        if k == 0.0 {
            res += consts::PI.powf(dimf / 2.0) / Gamma::gamma(dimf / 2.0) * (ms * ms).ln() / vol;
        } else {
            let y_squared = y.pow2().sum();
            res -= consts::PI.powf((2.0 + k) + (dimf / 2.0)) / Gamma::gamma(k + (dimf / 2.0))
                * (-1.0_f64).powf(k + 1.0)
                / Gamma::gamma(k + 1.0)
                * y_squared.powf(k)
                * (ms * ms).ln()
                / vol;
        }
    }
    res
}

/// calculates the Epstein zeta function.
///
/// * @param[in] nu: exponent for the Epstein zeta function.
/// * @param[in] dim: dimension of the input vectors.
/// * @param[in] a: matrix that transforms the lattice in the Epstein Zeta function.
/// * @param[in] x: x vector of the Epstein Zeta function.
/// * @param[in] y: y vector of the Epstein Zeta function.
/// * @return function value of the regularized Epstein zeta.
pub fn epstein_zeta(nu: f64, a: &Array2<f64>, x: &Array1<f64>, y: &Array1<f64>) -> Complex64 {
    epstein_zeta_internal(nu, a, x, y, 1.0, false)
}

/// calculates the regularized Epstein zeta function.
///
/// * @param[in] nu: exponent for the Epstein zeta function.
/// * @param[in] dim: dimension of the input vectors.
/// * @param[in] a: matrix that transforms the lattice in the Epstein Zeta function.
/// * @param[in] x: x vector of the Epstein Zeta function.
/// * @param[in] y: y vector of the Epstein Zeta function.
/// * @return function value of the regularized Epstein zeta.
pub fn epstein_zeta_reg(nu: f64, a: &Array2<f64>, x: &Array1<f64>, y: &Array1<f64>) -> Complex64 {
    epstein_zeta_internal(nu, a, x, y, 1.0, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // use approx::assert_abs_diff_eq;
    use manifest_dir_macros::path;
    use ndarray::array;
    use num_complex::ComplexFloat;
    use serde::Deserialize;

    #[test]
    fn test_sum_real() {
        let nu = 9.1366273707310508e-01;
        let lambda = 1.0;
        let z_arg_bound = 3.1172453105244717e+01;
        let m = array!([1.1547005383792517e+00, 0.0], [0.0, 8.6602540378443860e-01]);
        let x = array!(-1.1033540377972531e-01, 1.0725221992945301e-02);
        let y = array!(-8.6000420863103680e-02, 8.2532598838252230e-02);
        let cutoffs = array!(3, 4);
        let computed = sum_real(nu, lambda, &m, &x, &y, &cutoffs, z_arg_bound);
        let reference = c64(6.4776111947062631e+00, -6.9161715507768205e-03);
        let tolerance = 1.0e-15;
        // assert_abs_diff_eq!(reference.re, computed.re, epsilon = epsilon);
        // assert_abs_diff_eq!(reference.im, computed.im, epsilon = epsilon);
        let diff = reference - computed;
        let error_abs = diff.abs();
        let error_rel = error_abs / reference.abs();
        let error = if error_abs > error_rel {
            error_abs
        } else {
            error_rel
        };
        if error > tolerance {
            panic!();
        }
    }

    #[test]
    fn test_sum_fourier() {}

    #[derive(Debug, Deserialize)]
    struct ZetaRecord {
        nu_real: f64,
        nu_imag: f64,
        a1: f64,
        a2: f64,
        a3: f64,
        a4: f64,
        x1: f64,
        x2: f64,
        y1: f64,
        y2: f64,
        ref1: f64,
        ref2: f64,
    }

    #[test]
    fn test_epstein_zeta() {
        let path = path!("src", "tests", "csv", "epsteinZeta_Ref.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        let tol = 1.0e-13;
        let mut failed = 0;
        for result in rdr.deserialize() {
            let ZetaRecord {
                nu_real,
                nu_imag: _nu_imag,
                a1,
                a2,
                a3,
                a4,
                x1,
                x2,
                y1,
                y2,
                ref1,
                ref2,
            } = result.unwrap();
            let a = array!([a1, a2], [a3, a4]);
            let x = array!(x1, x2);
            let y = array!(y1, y2);
            let zeta_computed = epstein_zeta(nu_real, &a, &x, &y);
            let zeta_ref = c64(ref1, ref2);
            let diff = zeta_ref - zeta_computed;
            let error_abs = diff.abs();
            let error_rel = error_abs / zeta_ref.abs();
            let error = if error_abs > error_rel {
                error_abs
            } else {
                error_rel
            };
            if error >= tol {
                eprintln!(
                    "FAIL ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    zeta_ref.re, zeta_computed.re, error_abs, error_rel, tol
                );
                failed += 1;
            } else {
                println!(
                    "PASS ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    zeta_ref.re, zeta_computed.re, error_abs, error_rel, tol
                );
            }
        }
        if failed > 0 {
            panic!();
        }
    }

    #[test]
    fn test_epstein_zeta_reg() {
        let path = path!("src", "tests", "csv", "epsteinZetaReg_Ref.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        let tol = 1.0e-13;
        let mut failed = 0;
        for result in rdr.deserialize() {
            let ZetaRecord {
                nu_real,
                nu_imag: _nu_imag,
                a1,
                a2,
                a3,
                a4,
                x1,
                x2,
                y1,
                y2,
                ref1,
                ref2,
            } = result.unwrap();
            let a = array!([a1, a2], [a3, a4]);
            let x = array!(x1, x2);
            let y = array!(y1, y2);
            let zeta_computed = epstein_zeta_reg(nu_real, &a, &x, &y);
            let zeta_ref = c64(ref1, ref2);
            let diff = zeta_ref - zeta_computed;
            let error_abs = diff.abs();
            let error_rel = error_abs / zeta_ref.abs();
            let error = if error_abs > error_rel {
                error_abs
            } else {
                error_rel
            };
            if error >= tol {
                eprintln!(
                    "FAIL ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    zeta_ref.re, zeta_computed.re, error_abs, error_rel, tol
                );
                failed += 1;
            } else {
                println!(
                    "PASS ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    zeta_ref.re, zeta_computed.re, error_abs, error_rel, tol
                );
            }
        }
        if failed > 0 {
            panic!();
        }
    }
}
