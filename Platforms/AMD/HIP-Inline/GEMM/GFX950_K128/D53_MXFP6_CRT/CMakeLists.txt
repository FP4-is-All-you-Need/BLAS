cmake_minimum_required(VERSION 3.21)
project(awe_mxfp6 LANGUAGES CXX)

# ROCm tools. hipcc builds the HIP executables; the ROCm clang assembles the kernel code object.
# The HIP language support of CMake is not used so the recorded compiler arguments are passed as is.
set(AWE_HIP_ARCH "gfx950" CACHE STRING "AMD GPU architecture (MI350X)")
find_program(AWE_HIPCC hipcc HINTS "$ENV{ROCM_PATH}/bin" "$ENV{HIP_PATH}/bin" /opt/rocm/bin REQUIRED)
get_filename_component(_awe_rocm_bin "${AWE_HIPCC}" DIRECTORY)
get_filename_component(AWE_ROCM_ROOT "${_awe_rocm_bin}" DIRECTORY)
find_program(AWE_CLANG clang HINTS "${AWE_ROCM_ROOT}/llvm/bin" REQUIRED)
find_program(AWE_LLD ld.lld HINTS "${AWE_ROCM_ROOT}/llvm/bin" REQUIRED)
option(AWE_MXFP6_HIPBLASLT "Link hipBLASLt into the benchmark" ON)
option(AWE_ENABLE_GPU_TESTS "Register the GPU check as a test" OFF)

set(_awe_lib "${CMAKE_BINARY_DIR}/Lib")
set(_awe_bin "${CMAKE_BINARY_DIR}/Bin")
file(MAKE_DIRECTORY "${_awe_lib}" "${_awe_bin}")
set(_awe_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_awe_module "${_awe_lib}/awe_mxfp6_modular.co")

# Kernel code object from the AITER-derived assembly (THIRD_PARTY.md).
add_custom_command(OUTPUT "${_awe_module}"
    COMMAND "${AWE_CLANG}" -x assembler -target amdgcn-amd-amdhsa -mcpu=${AWE_HIP_ARCH} -mcode-object-version=6
        -c "${_awe_src}/AITER/modular.s" -o "${CMAKE_CURRENT_BINARY_DIR}/modular.o"
    COMMAND "${AWE_LLD}" -shared "${CMAKE_CURRENT_BINARY_DIR}/modular.o" -o "${_awe_module}"
    WORKING_DIRECTORY "${_awe_src}/AITER"
    DEPENDS "${_awe_src}/AITER/modular.s" "${_awe_src}/AITER/modular_descriptor.bin" "${_awe_src}/AITER/modular_note.bin"
    COMMENT "Assembling the MXFP6 GEMM code object for ${AWE_HIP_ARCH}"
    VERBATIM)
add_custom_target(awe_mxfp6_module ALL DEPENDS "${_awe_module}")

# Compiler arguments of the timings in the README; -greedy-reverse-local-assignment changes the register
# allocation of the encoder and CRT kernels.
set(_awe_hip_flags -O3 -std=c++20 --offload-arch=${AWE_HIP_ARCH} -mllvm -greedy-reverse-local-assignment=1 -fopenmp
    "-I${_awe_src}" "-DAWE_MXFP6_MODULE_PATH=\"${_awe_module}\"")
file(GLOB _awe_headers CONFIGURE_DEPENDS "${_awe_src}/*.h" "${_awe_src}/Tools/common.h")

function(awe_hip_executable name source)
    set(_output "${_awe_bin}/${name}")
    add_custom_command(OUTPUT "${_output}"
        COMMAND "${AWE_HIPCC}" ${_awe_hip_flags} ${ARGN} "${_awe_src}/${source}" "-L${AWE_ROCM_ROOT}/lib" -lrocblas -o "${_output}"
        DEPENDS "${_awe_src}/${source}" ${_awe_headers} "${_awe_module}"
        COMMENT "Compiling ${name} for ${AWE_HIP_ARCH}"
        VERBATIM)
    add_custom_target(${name} ALL DEPENDS "${_output}")
endfunction()

awe_hip_executable(awe_mxfp6_check Tests/gpu_check.hip)
if(AWE_MXFP6_HIPBLASLT)
    awe_hip_executable(awe_mxfp6_bench Tools/bench.hip -DAWE_MXFP6_HIPBLASLT=1 -lhipblaslt)
else()
    awe_hip_executable(awe_mxfp6_bench Tools/bench.hip)
endif()

# CPU test of the table, residues, CRT and rounding; needs no GPU or ROCm headers.
add_executable(awe_mxfp6_host_test Tests/host_exact_test.cpp)
target_compile_features(awe_mxfp6_host_test PRIVATE cxx_std_20)
target_include_directories(awe_mxfp6_host_test PRIVATE "${_awe_src}")
set_target_properties(awe_mxfp6_host_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_awe_bin}")

include(CTest)
if(BUILD_TESTING)
    add_test(NAME mxfp6_host_exact COMMAND awe_mxfp6_host_test)
    set_tests_properties(mxfp6_host_exact PROPERTIES LABELS cpu)
    if(AWE_ENABLE_GPU_TESTS)
        add_test(NAME mxfp6_gpu_check COMMAND "${_awe_bin}/awe_mxfp6_check")
        set_tests_properties(mxfp6_gpu_check PROPERTIES LABELS gpu TIMEOUT 1800)
    endif()
endif()
