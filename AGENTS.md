# AGENTS.md

Guidance for AI agents working in this repository.

## Cursor Cloud specific instructions

### What this repo is

The **Simd Library** is a native C/C++ image processing and ML library. There is no web app or long-running dev server. End-to-end validation is **CMake build → run `Test` (and optionally `Test.py`)** from `build/` with the repo root passed via `-r=..`.

### System packages (first-time / fresh VM)

Install before the first CMake configure:

```bash
sudo apt-get update
sudo apt-get install -y g++ libstdc++-13-dev build-essential cmake python3-numpy
```

The default `/usr/bin/c++` may be Clang without `libstdc++` linked; **use g++** for CMake (see configure below).

### Configure and build (from repo root)

```bash
cmake ./prj/cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc \
  -DSIMD_TOOLCHAIN="g++" \
  -DSIMD_TARGET="" \
  -DSIMD_AVX512VNNI=ON \
  -DSIMD_AMXBF16=ON \
  -DSIMD_TEST_FLAGS="-march=native" \
  -DSIMD_SHARED=ON

cmake --build build --parallel$(nproc)
```

Artifacts: `build/Test`, `build/libSimd.so`, and (when `SIMD_PYTHON=ON`) `build/Test.py` / `build/Simd.py`.

CMake may refresh `src/Simd/SimdVersion.h` when the detected version changes; that is normal.

### Run tests

From `build/`:

```bash
export LD_LIBRARY_PATH="$(pwd):$LD_LIBRARY_PATH"   # required when SIMD_SHARED=ON
./Test "-r=.." -fi=Sobel -tt=1 -ts=1              # quick smoke (C++)
python3 ./Test.py -r=.. -i Sobel                  # Python wrapper (uses -i/-e, not -fi/-fe)
```

Full CI-style C++ run (slow):

```bash
./Test "-r=.." -m=a -tt=$(nproc) "-ot=log.txt" -ts=5 -mt=1 -w=640 -h=480 -c=256
```

See `README.md` (Test Framework) for filters (`-fi`, `-fe`), modes (`-m=a` / `-m=s`), and options.

### Lint / format

No in-repo linter or formatter is configured. Quality gate is **compile + `Test`**.

### Optional

- **OpenCV**: `-DSIMD_OPENCV=ON` (system OpenCV dev packages required).
- **Docker**: `make -C prj/docker run` (optional containerized build).
- **Static library** (default): omit `-DSIMD_SHARED=ON`; `LD_LIBRARY_PATH` not needed; Python wrapper needs shared `libSimd.so`.
