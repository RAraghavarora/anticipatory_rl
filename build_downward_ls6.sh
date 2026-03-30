#!/bin/bash
# Build Fast Downward on Lonestar6 with a C++20-capable compiler.

set -euo pipefail

cd "$(dirname "$0")"

FD_GCC_MODULE="${FD_GCC_MODULE:-gcc/13.2.0}"

if ! command -v module >/dev/null 2>&1; then
  # "module" is usually a shell function on LS6, but keep the message explicit.
  echo "Environment modules are not available in this shell." >&2
  echo "Run this on LS6 after initializing the TACC module environment." >&2
  exit 1
fi

if [ -n "${FD_GCC_MODULE}" ]; then
  module load "${FD_GCC_MODULE}"
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ is not on PATH." >&2
  echo "Load a GCC module on LS6, for example via 'module avail gcc' then 'module load gcc/<version>'." >&2
  exit 1
fi

GXX_PATH=$(command -v g++)
GXX_VERSION=$(g++ -dumpfullversion -dumpversion)
GXX_MAJOR=${GXX_VERSION%%.*}

echo "Using g++: ${GXX_PATH}"
echo "g++ version: ${GXX_VERSION}"

if [ "${GXX_MAJOR}" -lt 10 ]; then
  echo "Fast Downward requires a C++20-capable compiler; current g++ is too old." >&2
  echo "Load GCC 10 or newer on LS6 and rebuild." >&2
  exit 1
fi

export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-${GXX_PATH}}"
GCC_LIBSTDCPP=$(g++ -print-file-name=libstdc++.so.6)
GCC_LIBDIR=$(dirname "${GCC_LIBSTDCPP}")
export LD_LIBRARY_PATH="${GCC_LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "CC=${CC}"
echo "CXX=${CXX}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

if [ "${FD_CLEAN_BUILD:-1}" = "1" ]; then
  echo "Removing cached Fast Downward build in downward/builds"
  rm -rf downward/builds
fi

(
  cd downward
  python build.py release
)

echo "Fast Downward build finished"
