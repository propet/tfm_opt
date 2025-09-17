# Project Setup with devenv

Linux only (on windows you can use WSL)

This project utilizes devenv to create a reproducible development environment, ensuring all dependencies and configurations are consistent.
1. Installation
First, you need to install devenv.
Since devenv is built on top of the Nix package manager, you must have Nix installed on your system.

Install Nix:
You can install Nix by running the following command in your terminal.
```
sh <(curl -L https://nixos.org/nix/install) --daemon
```

Install devenv:
Once Nix is installed, you can install devenv with this command:
```
nix-env --install --attr devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable
```

2. Environment Activation
With devenv installed, navigate to the root of this repository and run the following command to enter the development shell:
```
devenv shell
```
This command activates the environment defined in the devenv.nix file.

3. Python Package Management with uv
This project uses uv as the Python package manager.
The provided devenv.nix configuration is set up to automatically handle Python dependencies.
Upon entering the devenv shell, the uv sync command will be executed automatically.
This will install all the Python packages listed in your pyproject.toml file, making them immediately available within the development environment.
You will have access to the uv command-line interface for any further package management needs.


# Run python files

```
cd src/
python -m battery.battery_cant_sell # note no .py
```

# 💅 Formatting

Using [black](https://github.com/psf/black?tab=readme-ov-file) with maximum line-length of 120 characters, as specified in the `pyproject.toml` file.

At the root of the project, run:

```bash
black .
```
