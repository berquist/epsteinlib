# SPDX-FileCopyrightText: 2024 Jan Schmitz <schmitz@num.uni-sb.de>
#
# SPDX-License-Identifier: AGPL-3.0-only
{self, fenix, ...}: {
  perSystem = {
    pkgs,
    lib,
    self',
    ...
  }: let
    pythonDevEnv = pkgs.python3.withPackages (
      ps:
        with ps; [
          twine
          build
          mypy
          mpmath
          numpy
          matplotlib
          self'.packages.epsteinlib_python
          pylint
        ]
    );

    build_epstein = buildtype: (pkgs.stdenv.mkDerivation {
      name = "epsteinlib";
      src = self;
      outputs = [
        "out"
        "dev"
      ];

      nativeBuildInputs = with pkgs; [
        meson
        ninja
        pkg-config
        (pkgs.python3.withPackages (
          ps:
            with ps; [
              cython
            ]
        ))
      ];
      buildInputs = [];
      mesonBuildType = buildtype;
      mesonFlags = ["-Dbuild_python=false"];

      enableParallelBuilding = true;

      passthru.optional-dependencies.dev = with pkgs; [
        # Linters/formatters
        git
        doxygen_gui
        graphviz
        neovim
        gcovr
        pythonDevEnv
        clang-tools
        hugo
        go
      ];
      meta = {
        homepage = "https://github.com/epsteinlib/epsteinlib";
        license = with lib.licenses; [agpl3Only];
        mainProgram = "epsteinlib_c-lattice_sum";
      };
    });
    build_epstein_python = pkgs.python3Packages.buildPythonPackage rec {
      name = "epsteinlib";
      src = self;
      outputs = [
        "out"
        "dev"
      ];

      build-system = with pkgs; [
        python3Packages.meson-python
        python3Packages.cython
        pkg-config
      ];
      dependencies = with pkgs; [
        python3Packages.numpy
      ];
      pyproject = true;
      buildInputs = [];

      nativeCheckInputs = with pkgs; [
        python3Packages.unittestCheckHook
        python3Packages.mpmath
      ];

      unittestFlagsArray = [
        "-s"
        "python/tests"
        "-v"
      ];

      enableParallelBuilding = true;
      pythonImportsCheck = [name];

      meta = {
        homepage = "https://github.com/epsteinlib/epsteinlib";
        license = with lib.licenses; [agpl3Only];
      };
    };

    build_epstein_rust = pkgs.stdenv.mkDerivation rec {
      name = "epsteinlib_rust";
      src = self;

      buildInputs = with pkgs; [
        fenix.packages.${pkgs.system}.cargo
        fenix.packages.${pkgs.system}.rustc
      ];

      buildPhase = ''
        export CARGO_HOME=$TMPDIR/cargo-home
        cargo build --release
      '';

      installPhase = ''
        mkdir -p $out/lib
        cp target/release/libepsteinlib.* $out/lib/
        mkdir -p $out/include
        # Copy header files if they exist
        cp src/*.h $out/include/ 2>/dev/null || true
      '';

      meta = {
        homepage = "https://github.com/epsteinlib/epsteinlib";
        license = with lib.licenses; [agpl3Only];
      };
    };
  in {
    packages = rec {
      epsteinlib = build_epstein "release";
      epsteinlib_dbg = build_epstein "debug";
      epsteinlib_python = build_epstein_python;
      epsteinlib_rust = build_epstein_rust;
      default = epsteinlib;
      inherit pythonDevEnv;
    };
  };
}
