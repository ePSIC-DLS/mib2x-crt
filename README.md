# Container for MIB conversion

This repository contains scripts to build a container for MIB conversion at
ePSIC by using Podman and/or Buildah. You can use `podman --version` and
`buildah --version` to see if they are installed.

## Build with Podman and Buildah

By using the `build.sh` script, it uses Podman to first build the required
packages and then uses Buildah to copy only the necessary packages from the
builder image to a very small image, which does not contain a package manager.
This requires the host to have `dnf` installed, the package manager.

It is recommended to use this approach as it produces (almost) the smallest
image possible:

```console
buildah unshare ./build.sh
```

## Build with Podman only

You can also use Podman only to build the container. The image will be slightly
larger as the final base image has the package manager installed. This uses the
`Dockerfile` to build the image:

```console
podman build --format docker -t mib2x .
```

## Launch the container

You can find the container by

```console
podman images
```

To have an interactive shell of the container, you can do

```console
podman run --rm -it localhost/mib2x:latest sh
```

## Perform MIB conversion

To perform the MIB conversion, you can do

```console
podman run --rm -v <VISIT>:<VISIT> localhost/mib2x python mib_convert.py
--mib-path <MIB_FILE> --auto-reshape --ibf --create-json
```

The `-v` flag mounts the visit on the container. For example if you want to
convert `.mib` file in `/dls/e02/data/2024/cm37231-4/`:

```console
podman run --rm -v /dls/e02/data/2024/cm37231-4/:/dls/e02/data/2024/cm37231-4/
localhost/mib2x python mib_convert.py --mib-path
'/dls/e02/data/2024/cm37231-4/Merlin/WS2_n_fred_40um/20240821_150039/20240821_150035_data.mib'
--auto-reshape --ibf --create-json
```

If you want to run in debug mode, you can set the environment variable
`M2X_DEBUG` to 1 by using `-e 'M2X_DEBUG=1'` when you use `podman run`. This
will save the output file under `<VISIT>/processing/debug/` instead of
`<VISIT>/processing`.

For a full list of options, please refer to

```console
podman run --rm localhost/mib2x python mib_convert.py --help
```

```text
usage: mib_arg_parser.py [-h] --mib-path MIB_PATH [--auto-reshape | --no-reshaping | --use-fly-back | --known-shape] [--scan-x SCAN_X] [--scan-y SCAN_Y] [--ibf]
                         [--bin-sig-factor BIN_SIG_FACTOR] [--bin-nav-factor BIN_NAV_FACTOR] [--create-json] [--ptycho-config PTYCHO_CONFIG]
                         [--ptycho-template PTYCHO_TEMPLATE]

Convert MIB files to hdf5

options:
  -h, --help            show this help message and exit
  --mib-path MIB_PATH   path to the MIB file
  --auto-reshape        enable auto reshaping (default: True)
  --no-reshaping        disable reshaping (default: False)
  --use-fly-back        use fly-back method to reshape (default: False)
  --known-shape         use a known scan shape to reshape (default: False)
  --scan-x SCAN_X       number of pixels in x (columns) (default: 256)
  --scan-y SCAN_Y       number of pixels in y (rows) (default: 256)
  --ibf                 save integrated bright-field image
  --bin-sig-factor BIN_SIG_FACTOR
                        the pixel binning factor in the signal dimensions, 0 for no binning (default: 4)
  --bin-nav-factor BIN_NAV_FACTOR
                        the pixel binning factor in the navigation dimensions, 0 for no binning (default: 4)
  --create-json         create a PtyREX JSON config file
  --ptycho-config PTYCHO_CONFIG
                        the name of the PtyREX JSON config file (default: 'pty_recon')
  --ptycho-template PTYCHO_TEMPLATE
                        the path of the template PtyREX JSON config file (default: './UserExampleJson.json')
```
