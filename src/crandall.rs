// SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

use core::f64::consts;

use ndarray::Array1;
use num_complex::{c64, Complex64};
use special::Gamma;

use crate::gamma::{egf_gamma_star, egf_ugamma};

/// epsilon for the cutoff around arguments of the crandall function around zero.
const EPS_ZERO_PIY: f64 = consts::PI * 1.0e-64;

/// Calculates the regularization of the zero summand in the second sum in
/// Crandall's formula in the special case of nu = dim + 2k for some natural
/// number k.
///
/// * @param[in] s: dimension minus exponent of the regularized Epstein zeta function.
/// * @param[in] arg: input of the function
/// * @param[in] k: k = - s / 2 = (nu - d) / 2 as an integer
/// * @param[in] lambda: scaling parameter of crandalls formula
/// * @return arg ** (- s / 2) * (gamma(s / 2, arg) + ((-1)^k / k! ) * (log(arg) -
/// * log(lambda ** 2))
fn crandall_g_reg_nuequalsdimplus2k(s: f64, arg: f64, k: f64, lambda: f64) -> Complex64 {
    let mut g_reg = 0.0;
    let taylor_cutoff = 0.1 * 0.1 * consts::PI;
    // Taylor expansion if nu = dim and y close to zero.
    if s == 0.0 && arg < taylor_cutoff {
        let euler_gamma = 0.57721566490153286555;
        let taylor_coeffs = [
            -euler_gamma,
            1.0,
            -0.25,
            0.05555555555555555,
            -0.010416666666666666,
            0.0016666666666666668,
            -0.0002314814814814815,
            0.00002834467120181406,
            -3.1001984126984127e-6,
            3.0619243582206544e-7,
        ];
        for i in 0..taylor_coeffs.len() {
            g_reg += taylor_coeffs[i] * arg.powi(i as i32);
        }
    } else if arg == 0.0 {
        g_reg = 1.0 / k;
    } else {
        g_reg = arg.powf(k)
            * (egf_ugamma(-k, arg) + ((-1.0_f64).powf(k) / Gamma::gamma(k + 1.0)) * arg.log2());
    }
    // subtract polynomial of order k due to free parameter lambda
    g_reg -= arg.powf(k) * (lambda * lambda).log2();
    c64(g_reg, 0.0)
}

/// Calculates the regularization of the zero summand in the second sum in
/// Crandall's formula.
///
/// * @param[in] dim: dimension of the input vectors
/// * @param[in] s: dimension minus exponent of the regularized Epstein zeta function,
/// * that is d - nu
/// * @param[in] z: input vector of the function
/// * @param[in] prefactor: prefactor of the vector, e. g. lambda
/// * @return - gamma(s/2) * gammaStar(s/2, pi * prefactor * z**2),
/// * where gammaStar is the twice regularized lower incomplete gamma function if s is
/// * not equal to - 2k and (pi * prefactor * y ** 2) ** (- s / 2)
/// * (gamma(s / 2, pi * prefactor * z ** 2) + ((-1)^k / k! ) * (log(pi * y ** 2) -
/// * log(prefactor ** 2))) if s is  equal to - 2k for non negative natural number k
pub(crate) fn crandall_g_reg(s: f64, z: &Array1<f64>, prefactor: f64) -> Complex64 {
    let z_argument = z.dot(z) * consts::PI * prefactor * prefactor;
    let k = -(s / 2.0).round_ties_even();
    if s < 1.0 && (s == -2.0 * k) {
        crandall_g_reg_nuequalsdimplus2k(s, z_argument, k, prefactor)
    } else {
        c64(
            -Gamma::gamma(s / 2.0) * egf_gamma_star(s / 2.0, z_argument),
            0.0,
        )
    }
}

/// calculates bounds on when to use asymptotic expansion of the upper
/// incomplete gamma function, depending on the value of nu.
///
/// * @param[in] nu: exponent of the regularized Epstein zeta function.
/// * @return minimum value of z, when to use the fast asymptotic expansion in the
/// * calculation of the incomplete upper gamma function upperGamma(nu, z).
pub(crate) fn assign_z_arg_bound(nu: f64) -> f64 {
    if nu > 1.6 && nu < 4.4 {
        consts::PI * 2.99 * 2.99
    } else if nu > -3.0 && nu < 8.0 {
        consts::PI * 3.15 * 3.15
    } else if nu > -70.0 && nu < 40.0 {
        consts::PI * 3.35 * 3.35
    } else if nu > -600.0 && nu < 80.0 {
        consts::PI * 3.5 * 3.5
    } else {
        // do not use expansion if nu is too big
        f64::MAX
    }
}

/// Assumes x and y to be in the respective elementary lattice cell.
/// Multiply with exp(2 * PI * i * x * y) to get the second sum in Crandall's
///
/// * @param[in] nu: exponent of the regularized Epstein zeta function.
/// * @param[in] z: input vector of the function
/// * @param[in] prefactor: prefactor of the vector, e. g. lambda or 1/lambda in
/// *      Crandall's formula
/// * @return upperGamma(nu/2,pi prefactor * z**2)
/// *      / (pi * prefactor z**2)^(nu / 2) in
pub(crate) fn crandall_g(nu: f64, z: &Array1<f64>, prefactor: f64, z_arg_bound: f64) -> Complex64 {
    let z_argument = z.dot(z) * consts::PI * prefactor * prefactor;
    if z_argument < EPS_ZERO_PIY {
        c64(-2.0 / nu, 0.0)
    } else if z_argument > z_arg_bound {
        c64(
            (-z_argument).exp() * (-2.0 + 2.0 * z_argument + nu) / (2.0 * z_argument * z_argument),
            0.0,
        )
    } else {
        c64(
            egf_ugamma(nu / 2.0, z_argument) / z_argument.powf(nu / 2.0),
            0.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use manifest_dir_macros::path;
    use ndarray::arr1;
    use num_complex::ComplexFloat;
    use serde::Deserialize;

    #[test]
    fn test_assign_z_arg_bound() {
        #[derive(Debug, Deserialize)]
        struct Record {
            nu: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "assignzArgBound.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        for result in rdr.deserialize() {
            let Record {
                nu,
                reference,
                epsilon,
            } = result.unwrap();
            let computed = assign_z_arg_bound(nu);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_crandall_g() {
        #[derive(Debug, Deserialize)]
        struct Record {
            nu: f64,
            z1: f64,
            z2: f64,
            ref_real: f64,
            ref_imag: f64,
        }
        let path = path!("src", "tests", "csv", "crandall_g_Ref.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .comment(Some(b'#'))
            .from_path(path)
            .unwrap();
        let prefactor = 1.0;
        let tol = 7.4e-1;
        let mut failed = 0;
        for result in rdr.deserialize() {
            let record: Record = result.unwrap();
            let nu = record.nu;
            let z = arr1(&[record.z1, record.z2]);
            let z_arg_bound = assign_z_arg_bound(nu);
            let crandall_g_computed = crandall_g(nu, &z, prefactor, z_arg_bound);
            let crandall_g_ref = c64(record.ref_real, record.ref_imag);
            let diff = crandall_g_ref - crandall_g_computed;
            let error_abs = diff.abs();
            let error_rel = error_abs / crandall_g_ref.abs();
            let error = if error_abs > error_rel {
                error_abs
            } else {
                error_rel
            };
            if error >= tol {
                eprintln!(
                    "FAIL ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    crandall_g_ref.re, crandall_g_computed.re, error_abs, error_rel, tol
                );
                failed += 1;
            } else {
                println!(
                    "PASS ref: {:.16e} computed: {:.16e} error_abs: {:.16e} error_rel: {:.16e} tol: {:.16e}",
                    crandall_g_ref.re, crandall_g_computed.re, error_abs, error_rel, tol
                );
            }
        }
        if failed > 0 {
            panic!();
        }
    }
}
