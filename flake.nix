{
  description = "Shell for micromamba";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        fhs = pkgs.buildFHSUserEnv {
          name = "my-fhs-environment";

          targetPkgs = _: with pkgs;[
            pkgs.micromamba
            pkgs.gdb
            pkgs.valgrind
          ];

          profile = ''
            set -e
            eval "$(micromamba shell hook --shell=posix)"
            export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
            micromamba create -q -n my-mamba-environment
            micromamba activate my-mamba-environment
            micromamba install --yes -f conda-requirements.txt -c conda-forge
            set +e
            # zsh
          '';
        };
      in
      {
        devShell = fhs.env;

        # devShell = pkgs.mkShell {
        #   buildInputs = [
        #     pkgs.petsc
        #     pkgs.mpi
        #     pkgs.gcc
        #     pkgs.python3
        #     pkgs.gfortran
        #   ];
        # };
      }
    );
}
