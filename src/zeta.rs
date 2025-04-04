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
    // cuboid cutoffs
    let mut total_summands = 1;
    let mut total_cutoffs = Vec::new();
    for k in 0..m.ncols() {
        total_cutoffs.push(total_summands);
        total_summands *= 2 * cutoffs[k] + 1;
    }
    let mut sum = c64(0.0, 0.0);
    let mut epsilon = c64(0.0, 0.0);
    // First Sum (in real space)
    for n in 0..total_summands {
        let mut zv = Array1::default(m.ncols());
        for k in 0..m.ncols() {
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
    // second sum (in fourier space)
    // skips zero
    c64(0.0, 0.0)
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
    use serde::Deserialize;

    #[test]
    fn test_epsteinZeta_epsteinZetaReg() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "epsteinZeta_Ref.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        for result in rdr.deserialize() {
            let Record {
                a,
                x,
                reference,
                epsilon,
            } = result.unwrap();
            // let computed = epstein_zeta(nu, a, x, y);
            // assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }
}
