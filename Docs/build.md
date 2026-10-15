# CUDA build environment

The import comparison uses CUDA Toolkit **13.2.2**, containing nvcc **13.2.86**,
with CMake **3.31.6** and clang-format **20.1.8**.

Install the toolkit without a driver or administrator privileges:

```sh
sh cuda_13.2.2_595.71.05_linux.run --toolkit --installpath="$HOME/opt/cuda-13.2" --silent
uv venv "$HOME/opt/blas-tools/venv"
uv pip install --python "$HOME/opt/blas-tools/venv/bin/python" cmake==3.31.6 clang-format==20.1.8 numpy==2.2.6
export PATH="$HOME/opt/cuda-13.2/bin:$HOME/opt/blas-tools/venv/bin:$PATH"
nvcc --version
```

The official Linux x86_64 runfile is
[`cuda_13.2.2_595.71.05_linux.run`](https://developer.download.nvidia.com/compute/cuda/13.2.2/local_installers/cuda_13.2.2_595.71.05_linux.run).
Its published MD5 is `5d7a0bc1f719655f2083fc711cacf7cf`. Select the version from the
[NVIDIA archive](https://developer.nvidia.com/cuda-13-2-2-download-archive);
CUDA 13.2.0 and 13.2.1 contain different compiler builds.

## Build options and library placement

The [README](../README.md#build-and-minimal-use) gives the build commands.
Use separate build directories for K96 and K64. `AWE_CUDA_ARCH` defaults to
`sm_103a` for K96 and `sm_100a` for K64; other combinations are rejected.
No GPU query or kernel execution is needed to configure or build.

The build creates `Lib/libawe.so`, 1 shared library per imported engine,
and the selected geometry's Level-3 support libraries under `Lib/`. Keep these files together.
Build a single engine with `--target awe_Inline_k96_d17`, for example. CMake consumers
can link the corresponding imported target `awe::Inline_k96_d17`. Make the library
directory and the CUDA runtime visible to the dynamic loader.
The frontend `libawe.so` selects isolated modules at plan creation;
see [the C API](api.md).

The build uses the recorded nvcc argument arrays in
`Docs/Provenance/engines.json` and `additional_engines.json` directly, with the
output path and symbol-isolation linker flags changed. CMake's CUDA language is disabled
so its default flags and device-link rules cannot modify these recorded invocations.
The custom targets track all source/header dependencies.

The preparation host compiler is GCC 11.4. The measured K96 libraries were built
with GCC 13.3: device instructions match, while some weak host C++ symbols differ.
The linker hides implementation exports; the documented engine C exports are
retained. Neither host binary identity nor new performance measurements are claimed.

The frontend requires a C++17 compiler and the CUDA runtime. C callers need
`Include/AWE/awe.h` and link `-lawe`. CPU tests need Python 3 and NumPy;
the recorded run used NumPy 2.2.6. [Validation](validation.md) lists the tests.

Replay and CMake clear `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS` and `NVCC_CCBIN`.
Generated table headers use `clang-format off`.
Their generators are not included; see [generated tables](architecture.md#generated-tables).

## Engine loading and isolation

Each engine is linked with a unique SONAME, a version script exporting only its
recorded engine C interface, and local symbol binding (`-Bsymbolic`). The
loader uses `RTLD_NOW | RTLD_LOCAL`. Local binding matters even when another
library exports the same C names globally. C++ implementation symbols and CUDA
host registration details are hidden from the dynamic ABI. Kernel names and
CUDA translation-unit contents remain unchanged.

The frontend shares 1 loader object per canonical module path and holds that
module's mutex across create, execute, metadata/error access and destroy.
This serializes access to `__constant__` CRT memory and the module's
`crt_owner()` state. Each plan is bound to its creation device. Engine handles
stay resident for the process lifetime.

Loading and symbol lookup occur when opening an engine. Execution uses a cached
function pointer. The isolation tests use fake modules; CUDA call overhead was
not measured by those tests.
