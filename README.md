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
podman build -t mib2x .
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
