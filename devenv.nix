{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/basics/
  env = {
    NAME = "Pytorch Tests";
    HSA_OVERRIDE_GFX_VERSION = "11.0.0"; # adjust for your GPU if needed
    PYTORCH_ROCM_ARCH = "gfx1100"; # example for RX 6800 (Navi 21)
    MPLBACKEND = "TkAgg"; # matplotlib plots
  };

  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    gcc
    zlib
    zstd
    python311Packages.tkinter
  ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true; # activates automatically the environment
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $NAME
  '';

  enterShell = ''
    hello
    git --version
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
