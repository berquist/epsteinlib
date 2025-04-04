// SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Calculates the gamma function, the incomplete upper gamma function and the
//! regularized lower incomplete gamma function for evaluations of Crandall's
//! formula.
//!
//! * @see Walter Gautschi. “A Computational Procedure for Incomplete Gamma
//! Func-257 tions”. In: ACM Trans. Math. Softw. 5 (1979), pp.  466–481

use std::f64::consts;

use libm::ldexp;
use special::{Gamma, Primitive};

/// Choice of algorithm for the upper incomplete gamma function.
#[derive(Debug, PartialEq)]
enum Dom {
    Pt,
    Qt,
    Cf,
    Ua,
    Rek,
}

/// set type of algorithm to use depending on the parameters.
///
/// * @param[in] a: exponent of the upper incomplete gamma function.
/// * @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// * @return enum for the type of algorithm to use.
fn egf_domain(a: f64, x: f64) -> Dom {
    let alpha = if x >= 0.5 {
        x
    } else {
        (0.5_f64).ln() / (0.5 * x).ln()
    };
    if a <= alpha {
        if x <= 1.5 && a >= -0.5 {
            Dom::Qt
        } else if x <= 1.5 {
            Dom::Rek
        } else if a >= 12.0 && a >= x / 2.35 {
            Dom::Ua
        } else {
            Dom::Cf
        }
    } else if a >= 12.0 && x >= 0.3 * a {
        Dom::Ua
    } else {
        Dom::Pt
    }
}

fn egf_ldomain(a: f64, x: f64) -> Dom {
    let alpha = if x >= 0.5 {
        x
    } else {
        (0.5_f64).ln() / (0.5 * x).ln()
    };
    if a <= alpha {
        if x <= 1.5 && (a >= -0.5 || (a >= -0.75 && x <= ldexp(2.0, -15))) {
            Dom::Qt
        } else if x <= 1.5 {
            Dom::Rek
        } else if a >= 12.0 && a >= x / 2.35 {
            Dom::Ua
        } else {
            Dom::Cf
        }
    } else if a >= 12.0 && x >= 0.3 * a {
        Dom::Ua
    } else {
        Dom::Pt
    }
}

/// calculate upper gamma function with the recursion formula.
///
/// * @param[in] a: exponent of the upper incomplete gamma function.
/// * @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// * @return function value of the upper incomplete gamma function.
fn egf_pt(a: f64, x: f64) -> f64 {
    let mut sn = 1.0;
    let mut add = x / (a + 1.0);
    let mut i = 1;
    let EGF_EPS = ldexp(1.0, -54);
    while i < 80 && (add / sn).abs() >= EGF_EPS {
        sn += add;
        add *= x / (a + f64::from(i) + 1.0);
        i += 1;
    }
    sn * (-x).exp() / Gamma::gamma(a + 1.0)
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// * @param[in] a: exponent of the upper incomplete gamma function.
/// * @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// * @return function value of the upper incomplete gamma function.
fn egf_qt(a: f64, x: f64) -> f64 {
    let taylor = [
        -0.57721566490153286061,
        0.078662406618721020471,
        0.120665041652816256,
        -0.045873569729475233502,
        -0.003675835173930896754,
        0.0059461363539460768081,
        -0.0012728068927170227343,
        -0.00010763930085795762215,
        0.00010760237325699335067,
        -0.000020447909131122835485,
        -3.1305435033459682903e-7,
        9.3743913180807382831e-7,
        -1.9558810017362205406e-7,
        1.0045741524138656286e-8,
        3.9296464196572404677e-9,
        -1.0723612248119824624e-9,
        1.0891334567503768218e-10,
        4.5706745059276311356e-12,
        -3.2115889339774401184e-12,
        4.8521668466476558978e-13,
        -2.4820344080682008122e-14,
    ];
    let u = if a.abs() < 0.5 {
        let mut u1 = taylor[0];
        let mut f = 1.0;
        for i in 1..taylor.len() {
            f *= a;
            u1 += taylor[i] * f;
        } // u1 = g(a)
        let mut u2 = 0.0;
        let y = a * x.ln();
        f = 1.0;
        if y.abs() < 1.0 {
            for n in 1..=30 {
                f /= f64::from(n);
                u2 += f;
                f *= y;
            }
        } else {
            u2 = (y.exp() - 1.0) / y;
        }
        Gamma::gamma(1.0 + a) * (1.0 - a) * u1 - u2 * x.ln()
    } else {
        Gamma::gamma(a) - x.powf(a) / a
    };
    let mut v = 0.0;
    let mut f = 1.0;
    for i in 1..=30 {
        f *= -x / f64::from(i);
        v += f / (a + f64::from(i));
    }
    v *= -(x.powf(a));
    u + v
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// @param[in] a: exponent of the upper incomplete gamma function.
/// @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// @return function value of the upper incomplete gamma function.
fn egf_rek(a: f64, x: f64) -> f64 {
    let m = (0.5 - a).floor() as i32;
    let epsilon = a + f64::from(m);
    let mut g = egf_qt(epsilon, x) * x.exp() * x.powf(-epsilon);
    for n in 1..=m {
        g = 1.0 / (f64::from(n) - epsilon) * (1.0 - x * g);
    }
    g
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// @param[in] a: exponent of the upper incomplete gamma function.
/// @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// @return function value of the upper incomplete gamma function.
fn egf_cf(a: f64, x: f64) -> f64 {
    let mut s = 1.0;
    let mut rp = 1.0; // t_k-1
    let mut rv = 0.0; // rho_0
    let mut k = 1;
    // TODO
    let EGF_EPS = ldexp(1.0, -54);
    let bound = rp / s;
    let bound = (bound as f64).abs();
    while k <= 200 && bound >= EGF_EPS {
        let fk = f64::from(k);
        let ak = fk * (a - fk) / ((x + 2.0 * fk - 1.0 - a) * (x + 2.0 * fk + 1.0 - a));
        rv = -ak * (1.0 + rv) / (1.0 + ak * (1.0 + rv));
        rp *= rv;
        s += rp;
        k += 1;
    }
    s * x.powf(a) * (-x).exp() / (x + 1.0 - a)
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// @param[in] a: exponent of the upper incomplete gamma function.
/// @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// @return function value of the upper incomplete gamma function.
fn egf_ua_r(a: f64, eta: f64) -> f64 {
    let d = [
        1.0,
        -1.0 / 3.0,
        1.0 / 12.0,
        -2.0 / 135.0,
        1.0 / 864.0,
        1.0 / 2835.0,
        -139.0 / 777600.0,
        1.0 / 25515.0,
        -571.0 / 261273600.0,
        -281.0 / 151559100.0,
        8.29671134095308601e-7,
        -1.76659527368260793e-7,
        6.70785354340149857e-9,
        1.02618097842403080e-8,
        -4.38203601845335319e-9,
        9.14769958223679023e-10,
        -2.55141939949462497e-11,
        -5.83077213255042507e-11,
        2.43619480206674162e-11,
        -5.02766928011417559e-12,
        1.10043920319561347e-13,
        3.37176326240098538e-13,
        -1.39238872241816207e-13,
        2.85348938070474432e-14,
        -5.13911183424257258e-16,
        -1.97522882943494428e-15,
        8.09952115670456133e-16,
    ];
    let mut beta = Vec::new();
    beta.resize(26, 0.0);
    beta[25] = d[26];
    beta[24] = d[25];
    for n in (0..=23).rev() {
        // TODO
        beta[n] = (n + 2) as f64 * beta[n + 2] / a + d[n + 1];
    }
    let mut s = 0.0;
    let mut f = 1.0;
    for i in 0..=25 {
        s += beta[i] * f;
        f *= eta;
    }
    s *= a / (a + beta[1]);
    s * (-0.5 * a * eta * eta).exp() / (2.0 * consts::PI * a).sqrt()
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// @param[in] a: exponent of the upper incomplete gamma function.
/// @param[in] x: lower integral boundary of the upper incomplete gamma function.
/// @return function value of the upper incomplete gamma function.
fn egf_ua(a: f64, x: f64) -> f64 {
    let lambda = x / a;
    let eta = (2.0 * (lambda - 1.0 - lambda.ln())).sqrt();
    let eta = if lambda - 1.0 < 0.0 { -eta } else { eta };
    let ra = egf_ua_r(a, eta);
    (0.5 * Primitive::erfc(eta * (a / 2.0).sqrt())) + ra
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// * @param a: exponent of the upper incomplete gamma function.
/// * @param x: lower integral boundary of the upper incomplete gamma function.
/// * @return function value of the upper incomplete gamma function.
pub(crate) fn egf_ugamma(a: f64, x: f64) -> f64 {
    match egf_domain(a, x) {
        Dom::Pt => Gamma::gamma(a) * (1.0 - egf_pt(a, x) * x.powf(a)),
        Dom::Qt => egf_qt(a, x),
        Dom::Cf => egf_cf(a, x),
        Dom::Ua => Gamma::gamma(a) * egf_ua(a, x),
        Dom::Rek => (-x).exp() * x.powf(a) * egf_rek(a, x),
    }
}

/// calculate the upper incomplete gamma function as in Gautschi.
///
/// * @param a: exponent of the upper incomplete gamma function.
/// * @param x: lower integral boundary of the upper incomplete gamma function.
/// * @return function value of the upper incomplete gamma function.
pub(crate) fn egf_gamma_star(a: f64, x: f64) -> f64 {
    let EGF_EPS = ldexp(1.0, -54);
    if x.abs() < EGF_EPS {
        if a <= 0.1 && (a - a.round_ties_even()).abs() < EGF_EPS {
            0.0
        } else {
            1.0 / Gamma::gamma(a + 1.0)
        }
    } else {
        match egf_ldomain(a, x) {
            Dom::Pt => egf_pt(a, x),
            Dom::Qt => egf_pt(a, x),
            Dom::Cf => {
                if a <= 0.1 && (a - a.round_ties_even()).abs() < EGF_EPS {
                    x.powf(-a)
                } else {
                    (1.0 - egf_cf(a, x) / Gamma::gamma(a)) * x.powf(-a)
                }
            }
            Dom::Ua => (1.0 - egf_ua(a, x)) * x.powf(-a),
            Dom::Rek => {
                if a <= 0.1 && (a - a.round_ties_even()).abs() < EGF_EPS {
                    x.powf(-a)
                } else {
                    (1.0 - (-x).exp() * x.powf(a) * egf_rek(a, x) / Gamma::gamma(a)) * x.powf(-a)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use manifest_dir_macros::path;
    use serde::Deserialize;

    #[test]
    fn test_egf_domain() {}

    #[test]
    fn test_egf_ldomain() {}

    #[test]
    fn test_egf_pt() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_pt.csv");
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
            let computed = egf_pt(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_qt() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_qt.csv");
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
            let computed = egf_qt(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_rek() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_rek.csv");
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
            let computed = egf_rek(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_cf() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_cf.csv");
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
            let computed = egf_cf(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_ua_r() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            eta: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_ua_r.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        for result in rdr.deserialize() {
            let Record {
                a,
                eta,
                reference,
                epsilon,
            } = result.unwrap();
            let computed = egf_ua_r(a, eta);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_ua() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_ua.csv");
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
            let computed = egf_ua(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_ugamma() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_ugamma.csv");
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
            let computed = egf_ugamma(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }

    #[test]
    fn test_egf_gamma_star() {
        #[derive(Debug, Deserialize)]
        struct Record {
            a: f64,
            x: f64,
            reference: f64,
            epsilon: f64,
        }
        let path = path!("src", "tests", "csv", "egf_gammaStar.csv");
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
            let computed = egf_gamma_star(a, x);
            assert_abs_diff_eq!(reference, computed, epsilon = epsilon);
        }
    }
}
