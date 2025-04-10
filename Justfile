# SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

# Build and test the C code outside of Nix with Meson as the build system
default:
    meson setup build
    meson compile -C build
    ln -fsv build/compile_commands.json compile_commands.json
    meson test -v -C build

# Build and test the C code outside of Nix with CMake as the build system
cmake:
    cmake \
        -S {{justfile_directory()}} \
        -B {{justfile_directory()}}/build_cmake \
        -DCMAKE_INSTALL_PREFIX={{justfile_directory()}}/install_cmake \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DBUILD_SHARED_LIBS=1
    cmake --build {{justfile_directory()}}/build_cmake
    cmake --install {{justfile_directory()}}/build_cmake
    ctest --test-dir {{justfile_directory()}}/build_cmake

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

# Run cloc on the project
cloc:
    cloc --vcs=git {{justfile_directory()}}
