ARG IMAGE=ubuntu:bionic
FROM $IMAGE

RUN apt-get update && \
    apt-get -y install build-essential cmake rsync libx11-dev libxv-dev libxcb-shm0-dev gcc g++ subversion \
               curl wget unzip autoconf automake libtool zlib1g-dev git python && \
    apt-get clean

ENV CMAKE_VERSION=3.17.4
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

WORKDIR /simd
COPY . .
WORKDIR /simd/build
RUN cmake ../prj/cmake -DSIMD_SYNET=OFF -DSIMD_SHARED=OFF && make -j10