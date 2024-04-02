# syntax=docker/dockerfile:1
FROM flatironinstitute/triqs:3.1.1 as base
LABEL maintainer="Igor Krivenko <igor.s.krivenko@gmail.com>"
LABEL description="Implementation of the time-dependent dual TRILEX theory"
LABEL version="0.3.1"

USER root
RUN useradd -m -s /bin/bash -u 999 build && echo "build:build" | chpasswd
RUN usermod -aG sudo build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make g++-10 apt-utils vim vim-python-jedi

ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Download a C++20 compatible version of Boost
ARG BOOST_URL=https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
RUN curl -O -L $BOOST_URL && \
    tar -xf boost_1_78_0.tar.gz && \
    mv boost_1_78_0 /home/build/boost && \
    rm boost_1_78_0.tar.gz

USER build
RUN mkdir /home/build/realevol
WORKDIR /home/build/realevol

# Install ARPACK-NG
RUN git clone https://github.com/opencollab/arpack-ng.git arpack-ng.git
RUN mkdir arpack-ng.build
WORKDIR arpack-ng.build
RUN cmake ../arpack-ng.git                              \
        -DCMAKE_INSTALL_PREFIX=/usr                     \
        -DCMAKE_BUILD_TYPE=Release                      \
        -DBUILD_SHARED_LIBS=ON                          \
        -DICB=ON                                        \
        -DMPI=ON
RUN make -j6 && ctest --output-on-failure
USER root
RUN make install

# Install realevol
USER build
WORKDIR /home/build/realevol

RUN mkdir -p -m 0700 /home/build/.ssh && \
    ssh-keyscan github.com >> /home/build/.ssh/known_hosts
RUN --mount=type=ssh,uid=999 \
    git clone -v git@github.com:krivenko/triqs-realevol.git realevol.git
RUN mkdir realevol.build
WORKDIR realevol.build
RUN cmake ../realevol.git                               \
        -DCMAKE_INSTALL_PREFIX=/usr                     \
        -DCMAKE_BUILD_TYPE=Release                      \
        -DBoost_INCLUDE_DIR=$HOME/boost                 \
        -Darpack-ng_DIR=/usr/lib/x86_64-linux-gnu/cmake \
        -DBUILD_SHARED_LIBS=ON                          \
        -DBuild_Tests=ON                                \
        -DBuild_Benchmarks=OFF
RUN make VERBOSE=1 -j6 && ctest --output-on-failure
USER root
RUN make install

# Cleanup build files
USER root
RUN rm -rf /home/build/boost /home/build/realevol

# Install TDDT
USER build
COPY --chown=build . /src/tddt
WORKDIR /src/tddt
RUN pip3 install --user -r requirements.txt
RUN pip3 install --user .

# Test TDDT
RUN python3 -m pytest -v --with-mpi
