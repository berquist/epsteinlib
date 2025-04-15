// SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

use std::f64::consts;

use ndarray::{Array1, Array2};
use num_complex::{c64, Complex64};

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
    cutoffs: &[usize],
    z_arg_bound: f64,
) -> Complex64 {
    let dim = m.ncols();
    let mut zv = Array1::default(dim);
    // cuboid cutoffs
    let mut total_summands = 1;
    let mut total_cutoffs = Vec::new();
    for k in 0..dim {
        total_cutoffs.push(total_summands);
        total_summands *= 2 * cutoffs[k] + 1;
    }
    let mut sum = c64(0.0, 0.0);
    let mut epsilon = c64(0.0, 0.0);
    // First Sum (in real space)
    for n in 0..total_summands {
        for k in 0..dim {
            zv[k] = (((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k]) as f64;
        }
        let mut lv = m.dot(&zv);
        let rot = c64(0.0, -2.0 * consts::PI * lv.dot(y));
        lv -= x;
        // summing using Kahan's method
        let auxy = rot * crandall::crandall_g(nu, &lv, 1.0 / lambda, z_arg_bound) - epsilon;
        let auxt = sum + auxy;
        epsilon = (auxt - sum) - auxy;
        sum = auxt;
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
    cutoffs: &[usize],
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
            zv[k] = (((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k]) as f64;
        }
        let lv = m_invt.dot(&zv) + y;
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
            zv[k] = (((n / total_cutoffs[k]) % (2 * cutoffs[k] + 1)) - cutoffs[k]) as f64;
        }
        let lv = m_invt.dot(&zv) + y;
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
/// * is needet.
/// * @return projection of v to the elementary lattice cell.
fn vector_proj(m: &Array2<f64>, m_invt: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut todo = false;
    let mut vt = Array1::zeros(v.len());
    vt
}

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
    // 1. Transform: Compute determinant and fourier transformed matrix, scale
    // both of them
    c64(0.0, 0.0)
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

    use manifest_dir_macros::path;
    use ndarray::array;
    use num_complex::ComplexFloat;
    use serde::Deserialize;

    #[test]
    fn test_epstein_zeta() {
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
}
