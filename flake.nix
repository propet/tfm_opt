{
  description = "Shell for micromamba";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-22 = {
      url = "github:nixos/nixpkgs/nixos-22.05";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-22,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pkgs-22 = import nixpkgs-22 {
          inherit system;
          config.allowUnfree = true;
        };

        guiPkgs = with pkgs; [
          # Qt packages
          qt6.qtbase
          qt6.qtwayland

          # Wayland and session management
          wayland
          wayland-protocols
          wayland-utils
          pipewire
          xdg-utils

          # Core graphics libraries and drivers
          mesa
          libxkbcommon

          # X11 compatibility libraries (for XWayland)
          xorg.libX11
          xorg.libXcursor
          xorg.libXrandr
          xorg.libXi
          xorg.libXinerama
          xorg.libXcomposite
          xorg.libXdamage
          xorg.libXfixes
          xorg.libXrender
          xorg.xcbutilcursor
        ];

        fhs = pkgs.buildFHSUserEnv {
          name = "my-fhs-environment";

          targetPkgs =
            _:
            with pkgs;
            [
              micromamba
              gdb
              valgrind
              mpi
              blas
              lapack
              metis
            ]
            ++ guiPkgs; # Append the gui packages

          profile = ''
            set -e

            # 1. Set up and activate the micromamba environment first.
            #    This will heavily modify the environment.
            eval "$(micromamba shell hook --shell=posix)"
            export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba

            # Make environment creation idempotent to speed up subsequent shell loads
            if [ ! -d "$MAMBA_ROOT_PREFIX/envs/my-mamba-environment" ]; then
              echo "--- Creating Mamba environment 'my-mamba-environment'..."
              micromamba create -q -n my-mamba-environment
              micromamba install --yes -f conda-requirements.txt -c conda-forge -n my-mamba-environment
            fi

            micromamba activate my-mamba-environment


            # 2. NOW, after Mamba is active, we forcefully prepend our Nix paths.
            #    This ensures they are found before any Mamba-installed libraries.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath guiPkgs}:$LD_LIBRARY_PATH"
            export QT_PLUGIN_PATH="${
              pkgs.lib.makeSearchPath "lib/qt-6/plugins" [
                pkgs.qt6.qtbase
                pkgs.qt6.qtwayland
              ]
            }"
            export QT_QPA_PLATFORM="wayland;xcb"

            # 3. Add some debugging output to verify the paths in your shell
            echo "--- Nix FHS Debug Info ---"
            echo "Final QT_PLUGIN_PATH=$QT_PLUGIN_PATH"
            echo "--------------------------"

            set +e
            # zsh
          '';
        };
      in
      {
        devShell = fhs.env;

        # devShell = pkgs.mkShell {
        #   # OS provides conda
        #
        #   buildInputs = with pkgs; [
        #     marp-cli # presentations
        #     pkgs-22.petsc
        #     mpi
        #     gcc
        #     gfortran
        #   ];
        #
        #   shellHook = ''
        #     # Conda setup
        #     __conda_setup="$('/home/luis/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
        #     if [ $? -eq 0 ]; then
        #         eval "$__conda_setup"
        #     else
        #         if [ -f "/home/luis/miniconda3/etc/profile.d/conda.sh" ]; then
        #             . "/home/luis/miniconda3/etc/profile.d/conda.sh"
        #         else
        #             export PATH="/home/luis/miniconda3/bin:$PATH"
        #         fi
        #     fi
        #     unset __conda_setup
        #
        #     conda env create -f conda-requirements.yml
        #     conda activate myenv
        #   '';
        # };
      }
    );
}
