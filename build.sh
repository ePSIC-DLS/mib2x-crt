#!/usr/bin/env bash
# Pre-requisites
#
# podman and Buildah
#
# This script builds the container image for mib file conversion.
#
# First stage:
#
#   Use podman and the Dockerfile to build artefacts
#
# Second stage:
#
#   Use Buildah to install runtime dependencies and
#   copy the arfefacts from the first stage to ubi micro
set -ex

main() {
    # install and build everything using podman
    podman build --target builder -t mib2x-builder .

    # pull the ubi micro image
    microcontainer=$(buildah from redhat/ubi8-micro:latest)

    # mount container's root file system
    micromount=$(buildah mount ${microcontainer})

    # install run-time dependencies
    # dnf run as non-root so it won't have permission to
    # write log/read cache on the host
    # See: https://bugzilla.redhat.com/show_bug.cgi?id=1687523#c15
    dnf install \
        --installroot "${micromount}" \
        --releasever 8 \
        --setopt install_weak_deps=false \
        --nodocs -y \
        python3.12 \
        libgomp \
        libstdc++
    dnf clean all --installroot "${micromount}"

    # unmount container's root file system
    buildah umount "${microcontainer}"

    # copy artefacts from builder
    buildah copy --from mib2x-builder "${microcontainer}" \
        /usr/local /usr/local

    buildah copy --from mib2x-builder "${microcontainer}" \
        /opt/mib_props/*.so /opt/fast_bin_stash/*.so /usr/local/lib/python3.12/site-packages/

    buildah copy --from mib2x-builder "${microcontainer}" \
        /etc/passwd /etc/passwd

    buildah copy --from mib2x-builder "${microcontainer}" \
        /etc/group /etc/group

    buildah copy --from mib2x-builder --chown=ruska "${microcontainer}" \
        /home/ruska /home/ruska

    # executing script
    buildah copy --chown=ruska --chmod=644 "${microcontainer}" \
        import_test.py mib_convert.py UserExampleJson.json /home/ruska/

    # set config
    # not set user as 'ruska' for flexibility
    buildah config \
        --author='Timothy Poon (timothy.poon@diamond.ac.uk)' \
        --workingdir='/home/ruska' \
        "${microcontainer}"

    # save the image
    buildah commit $microcontainer mib2x
}

main $@
