# SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

default:
    meson setup build
    meson compile -C build
    ln -fsv build/compile_commands.json compile_commands.json
    meson test -v -C build

rust:
    cargo install cargo-tarpaulin
    cargo tarpaulin --out Xml --out Html --workspace

nix:
    nix build -L
    nix run
    nix build -L .#epsteinlib_dbg
    nix run .#epsteinlib_dbg
    nix flake check -L
    nix develop -c tests
    nix develop -c docs
