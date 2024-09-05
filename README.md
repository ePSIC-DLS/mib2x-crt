# Build container for MIB conversion

This repository contains scripts to build a container for MIB conversion by
using podman and Buildah. You can use `podman --version` and `buildah
--version` to see if they are installed.

To build the container:

```console
buildah unshare ./build.sh
```

You can find the container by

```console
podman images
```

To have an interactive shell of the container, you can do

```console
podman run --rm -it localhost/mib2x:latest sh
```
