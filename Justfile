# SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

# Build and test the C code outside of Nix
default:
    meson setup build
    meson compile -C build
    ln -fsv build/compile_commands.json compile_commands.json
    meson test -v -C build

# Test the Rust code with code coverage outside of Nix
rust:
    cargo install cargo-tarpaulin
    cargo tarpaulin --out Xml --out Html --workspace

# Build and test the code within Nix
nix:
    nix build -L
    nix run
    nix build -L .#epsteinlib_dbg
    nix run .#epsteinlib_dbg
    nix flake check -L
    nix develop -c tests
    nix develop -c docs
