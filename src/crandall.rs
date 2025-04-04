// SPDX-FileCopyrightText: 2024 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

use std::f64::consts;

use num_complex::{c64, Complex64};

const EPS_ZERO_Y: f64 = 1.0e-64;
const EPS_ZERO_PIY: f64 = consts::PI * 1.0e-64;

fn crandall_gReg_nuequalsdim(s: f64, arg: f64, k: f64, lambda: f64) -> Complex64 {
    let mut gReg = c64(0.0, 0.0);
    let taylorCutoff = 0.1 * 0.1 * consts::PI;
    // Taylor expansion if nu = dim and y close to zero.
    if s == 0.0 && arg < taylorCutoff {
    } else if arg == 0.0 {
        // gReg = 1.0 / k;
    } else {
        // gReg = arg.powf(k) * (egf_ugamma(-k, arg) + (-1.0.powf(k) / (k + 1.0).gamma()) * arg.log2());
    }
    // subtract polynomial of order k due to free parameter lambda
    gReg
}

// fn crandall_gReg(dim: usize, nu: f64, z: &[f64], prefactor: f64) -> Complex64 {
//     let zArgument = consts::PI * prefactor * prefactor;
//     if zArgument < EPS_ZERO_PIY {
//     } else if zArgument > zArgBound {
//     } else {
//         // -zArgument.e
//     }
// }

fn assignzArgBound(nu: f64) -> f64 {
    if nu > 1.6 && nu < 4.4 {
        consts::PI * 2.99 * 2.99
    } else if nu > -3.0 && nu < 8.0 {
        consts::PI * 3.15 * 3.15
    } else if nu > -70.0 && nu < 40.0 {
        consts::PI * 3.35 * 3.35
    } else if nu > -600.0 && nu < 80.0 {
        consts::PI * 3.5 * 3.5
    } else {
        f64::MAX
    }
}

fn crandall_g(dim: usize, nu: f64, z: &[f64], prefactor: f64, zArgBound: f64) -> Complex64 {
    c64(0.0, 0.0)
}
