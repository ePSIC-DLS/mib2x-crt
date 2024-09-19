FROM redhat/ubi8:latest as builder

ENV PIP_NO_CACHE_DIR=1 \
    HDF5_VERSION='1.14.3' \
    NUMPY_VERSION='2.1.1' \
    MIB_PROPS_VERSION='1.0.1'

RUN dnf install --disablerepo="*" \
    --enablerepo="ubi-8-baseos-rpms" \
    --enablerepo="ubi-8-appstream-rpms" \
    --enablerepo="ubi-8-codeready-builder-rpms" \
    --setopt install_weak_deps=false \
    --nodocs -y \
    gcc \
    gcc-c++ \
    make \
    wget \
    tar \
    zlib-devel \
    git-core \
    libjpeg-devel \
    python3.12 \
    python3.12-pip \
    python3.12-wheel \
    python3.12-setuptools \
    python3.12-devel \
    python3.12-Cython \
    && dnf clean all

WORKDIR /opt

RUN wget --quiet https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-${HDF5_VERSION//./_}.tar.gz \
    && gzip -cd hdf5-${HDF5_VERSION//./_}.tar.gz | tar xf - \
    && mv hdf5-hdf5-${HDF5_VERSION//./_} hdf5-${HDF5_VERSION}

# not sure how to disable examples in Autotools so just put them to /tmp and
# they are not copied over, this is easier
RUN cd hdf5-${HDF5_VERSION} \
    && CFLAGS="-mavx2" \
    ./configure \
   --prefix=/usr/local \
   --enable-build-mode=production \
   --enable-shared \
   --disable-static \
   --disable-fortran \
   --disable-cxx \
   --disable-tools \
   --disable-tests \
   --with-examplesdir=/tmp && \
   make -j$(nproc) && make install

# CMake (need install) can disable examples, but the optimization level is not
# high even this is Release, so not use it but worth keep it for later
# investigation
#RUN cd hdf5-${HDF5_VERSION} \
    #&& mkdir -p build \
    #&& cd build \
    #&& cmake \
    #-G "Unix Makefiles" \
    #-DCMAKE_C_FLAGS:STRING="-mavx2" \
    #-DCMAKE_INSTALL_PREFIX:STRING=/usr/local \
    #-DZLIB_DIR:STRING=/usr/lib64/ \
    #-DCMAKE_BUILD_TYPE:STRING=Release \
    #-DBUILD_STATIC_LIBS:BOOL=OFF \
    #-DHDF5_BUILD_TOOLS:BOOL=OFF \
    #-DBUILD_TESTING:BOOL=OFF \
    #-DHDF5_BUILD_EXAMPLES:BOOL=OFF \
    #.. \
    #&& make -j$(nproc) && make install

# numpy without blas
RUN git clone https://github.com/numpy/numpy.git \
    && cd numpy \
    && git switch --detach v"${NUMPY_VERSION}" \
    && git submodule update --init \
    && python3.12 -m pip install . -v \
       -Csetup-args="-Dblas=none" \
       -Csetup-args="-Dlapack=none" \
       -Csetup-args="-Dallow-noblas=true"

# remove other hdf5plugin filter after installation, only need h5blosc
# silent log when fail to initialise other filters
RUN python3.12 -m pip install \
    blosc \
    && HDF5_DIR='/usr/local' \
    python3.12 -m pip install \
    --no-binary=h5py --no-build-isolation \
    h5py hdf5plugin \
    && find /usr/local/lib64/python3.12/site-packages/hdf5plugin/plugins/ \
       -type f -not -name "libh5blosc.so" -exec rm {} + \
    && sed -i '/Cannot initialize filter/s/^/#/' \
       /usr/local/lib64/python3.12/site-packages/hdf5plugin/_utils.py

# PIL with jpeg support
RUN python3.12 -m pip install --no-binary=Pillow Pillow \
    -C parallel=$(nproc) \
    -C zlib=disable \
    -C jpeg=enable \
    -C tiff=disable \
    -C freetype=disable \
    -C raqm=disable \
    -C lcms=disable \
    -C webp=disable \
    -C webpmux=disable \
    -C jpeg2000=disable \
    -C imagequant=disable \
    -C xcb=disable

# build mib_props C extension
RUN git clone https://github.com/ePSIC-DLS/mib_props.git \
    && cd mib_props \
    && git switch --detach v${MIB_PROPS_VERSION} \
    && python3.12 setup.py build_ext --inplace

# build fast_bin C extension
RUN git clone https://github.com/ptim0626/fast_bin_stash.git \
    && cd fast_bin_stash \
    && python3.12 setup.py build_ext --inplace

# install stripped-down HyperSpy/rsciio
COPY hspy_stripped hspy_stripped

RUN python3.12 -m pip install ./hspy_stripped

# add a non-root user
RUN useradd --user-group --create-home ruska

#===============================================================================
FROM redhat/ubi8-minimal:latest

RUN microdnf install --disablerepo="*" \
    --enablerepo="ubi-8-baseos-rpms" \
    --enablerepo="ubi-8-appstream-rpms" \
    --enablerepo="ubi-8-codeready-builder-rpms" \
    --setopt install_weak_deps=false \
    --nodocs -y \
    python3.12 \
    libgomp \
    libstdc++ \
    libjpeg \
    && microdnf clean all \
    && rm -rf /usr/share/doc /usr/share/man /usr/share/info

COPY --from=builder /usr/local /usr/local

COPY --from=builder /opt/mib_props/*.so /opt/fast_bin_stash/*.so /usr/local/lib/python3.12/site-packages/

COPY --from=builder /etc/passwd /etc/passwd

COPY --from=builder /etc/group /etc/group

COPY --from=builder /home/ruska /home/ruska

COPY --chown=ruska --chmod=644 mib_convert.py UserExampleJson.json /home/ruska/

ENV PYTHONUNBUFFERED=1

LABEL description='This image converts a .mib file to .hdf5/.hspy alongside other processing such as pixel binning at ePSIC in Diamond Light Source. To build the container, see https://github.com/ePSIC-DLS/mib2x-crt'

WORKDIR /home/ruska
