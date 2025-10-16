{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        devShell = with pkgs;
          mkShell {
            hardeningDisable = [ "all" ];
            buildInputs = [
              python_312
            ];
          };
      });
}
