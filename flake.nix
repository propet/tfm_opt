{
  description = "Shell for micromamba";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-22 = {
      url = "github:nixos/nixpkgs/nixos-22.05";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-22, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pkgs-22 = import nixpkgs-22 {
          inherit system;
          config.allowUnfree = true;
        };
        fhs = pkgs.buildFHSUserEnv {
          name = "my-fhs-environment";

          targetPkgs = _: with pkgs; [
            pkgs.micromamba
            pkgs.gdb
            pkgs.valgrind
            pkgs.mpi
            pkgs.blas
            pkgs.lapack
            pkgs.metis
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
        # devShell = fhs.env;

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.marp-cli  # presentations
            pkgs-22.petsc
            pkgs.mpi
            pkgs.gcc
            pkgs.gfortran
          ];

          shellHook = ''
            # Conda setup
            __conda_setup="$('/home/luis/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
            if [ $? -eq 0 ]; then
                eval "$__conda_setup"
            else
                if [ -f "/home/luis/miniconda3/etc/profile.d/conda.sh" ]; then
                    . "/home/luis/miniconda3/etc/profile.d/conda.sh"
                else
                    export PATH="/home/luis/miniconda3/bin:$PATH"
                fi
            fi
            unset __conda_setup

            conda env create -f conda-requirements.yml
            conda activate myenv
          '';
        };
      }
    );
}
