// SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
//
// SPDX-License-Identifier: AGPL-3.0-only

use epsteinlib::zeta::epstein_zeta;
use ndarray::array;

#[test]
fn lattice_sum() {
    let madelung_ref = -1.7475645946331821906362120355443974;
    let m = array!([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]); // identity matrix for whole numbers
    let x = array!(0.0, 0.0, 0.0); // no shift
    let y = array!(0.5, 0.5, 0.5); // alternating sum
    let nu = 1.0;
    let madelung = epstein_zeta(nu, &m, &x, &y).re;
    println!("Madelung sum in 3 dimensions:\t {:.16}", madelung);
    println!("Reference value:\t\t {:.16}", madelung_ref);
    println!(
        "Relative error:\t\t\t {:.2e}",
        (madelung_ref - madelung).abs() / madelung_ref.abs()
    );
    assert!((madelung - madelung_ref).abs() < (1.0e1_f64).powi(-14));
}
