cmake_minimum_required(VERSION 3.31)
project(awe LANGUAGES CXX)

add_library(awe_engine_loader STATIC Routines/Common/engine_loader.cpp)
target_compile_features(awe_engine_loader PUBLIC cxx_std_17)
set_target_properties(awe_engine_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(awe_engine_loader PUBLIC ${CMAKE_DL_LIBS})
target_include_directories(awe_engine_loader PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Routines/Common")

set(AWE_BACKEND "K96" CACHE STRING "Fixed engine geometry: K96 or K64")
set_property(CACHE AWE_BACKEND PROPERTY STRINGS K96 K64)
if(AWE_BACKEND STREQUAL "K96")
    set(_awe_k 96)
    set(_awe_arch sm_103a)
elseif(AWE_BACKEND STREQUAL "K64")
    set(_awe_k 64)
    set(_awe_arch sm_100a)
else()
    message(FATAL_ERROR "AWE_BACKEND must be K96 or K64")
endif()
set(AWE_CUDA_ARCH "${_awe_arch}" CACHE STRING "CUDA architecture for the selected geometry")
if(NOT AWE_CUDA_ARCH STREQUAL _awe_arch)
    message(FATAL_ERROR "${AWE_BACKEND} requires ${_awe_arch}; ${AWE_CUDA_ARCH} is unsupported")
endif()
find_program(AWE_NVCC nvcc REQUIRED)
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS --unset=NVCC_CCBIN
    "${AWE_NVCC}" --version
    OUTPUT_VARIABLE _awe_version RESULT_VARIABLE _awe_version_result)
if(NOT _awe_version_result EQUAL 0 OR NOT _awe_version MATCHES "V13\\.2\\.86")
    message(FATAL_ERROR "The recorded source comparison requires nvcc 13.2.86")
endif()

# Use the recorded invocation directly: CMake's CUDA language rules add flags
# that are absent from the reference build. Each library keeps its original ABI.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/engines.json" _awe_engines)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/engines.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/additional_engines.json")
# Additional imports keep separate source and measurement records.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/additional_engines.json" _awe_additional)
string(JSON _awe_more_count LENGTH "${_awe_additional}")
if(_awe_more_count GREATER 0)
    math(EXPR _awe_more_last "${_awe_more_count} - 1")
    foreach(_awe_more_index RANGE ${_awe_more_last})
        string(JSON _awe_next LENGTH "${_awe_engines}")
        string(JSON _awe_entry GET "${_awe_additional}" ${_awe_more_index})
        string(JSON _awe_engines SET "${_awe_engines}" ${_awe_next} "${_awe_entry}")
    endforeach()
endif()
string(JSON _awe_count LENGTH "${_awe_engines}")
math(EXPR _awe_last "${_awe_count} - 1")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Lib")
option(AWE_BUILD_TEMPLATE "Build the device-equivalent Template engines for comparison" OFF)
foreach(_awe_index RANGE ${_awe_last})
    string(JSON _awe_engine_k GET "${_awe_engines}" ${_awe_index} k)
    if(NOT _awe_engine_k EQUAL _awe_k)
        continue()
    endif()
    string(JSON _awe_id GET "${_awe_engines}" ${_awe_index} id)
    string(JSON _awe_path GET "${_awe_engines}" ${_awe_index} path)
    string(JSON _awe_argc LENGTH "${_awe_engines}" ${_awe_index} arguments)
    math(EXPR _awe_arglast "${_awe_argc} - 1")
    set(_awe_args)
    foreach(_awe_argindex RANGE ${_awe_arglast})
        string(JSON _awe_arg GET "${_awe_engines}" ${_awe_index} arguments ${_awe_argindex})
        list(APPEND _awe_args "${_awe_arg}")
    endforeach()
    set(_awe_source "${CMAKE_CURRENT_SOURCE_DIR}/${_awe_path}")
    set(_awe_output "${CMAKE_CURRENT_BINARY_DIR}/Lib/libawe_${_awe_id}.so")
    set(_awe_exports "${CMAKE_CURRENT_SOURCE_DIR}/CMake/engine_exports.map")
    if(_awe_id STREQUAL "k64_z53_general")
        set(_awe_exports "${CMAKE_CURRENT_SOURCE_DIR}/CMake/complex_adapter_exports.map")
    endif()
    if(_awe_id MATCHES "^k64_d(53|57)_trsm$")
        set(_awe_exports "${CMAKE_CURRENT_SOURCE_DIR}/CMake/trsm_exports.map")
    endif()
    if(_awe_id MATCHES "_i8$")
        set(_awe_exports "${CMAKE_CURRENT_SOURCE_DIR}/CMake/integer_exports.map")
    endif()
    list(APPEND _awe_args "-Xlinker=-Bsymbolic"
        "-Xlinker=--version-script=${_awe_exports}"
        "-Xlinker=-soname=libawe_${_awe_id}.so")
    file(GLOB_RECURSE _awe_dependencies CONFIGURE_DEPENDS
        "${_awe_source}/*.cu" "${_awe_source}/*.cuh" "${_awe_source}/*.h"
        "${_awe_source}/*.hpp" "${_awe_source}/*.inc")
    if(EXISTS "${_awe_source}/l8rt.cu")
        list(APPEND _awe_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/Platforms/NVIDIA/CUDA-Common/Common/additional_options.h")
    endif()
    add_custom_command(OUTPUT "${_awe_output}"
        COMMAND "${CMAKE_COMMAND}" -E env --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS --unset=NVCC_CCBIN
            "${AWE_NVCC}" ${_awe_args} -o "${_awe_output}"
        WORKING_DIRECTORY "${_awe_source}"
        DEPENDS ${_awe_dependencies} "${CMAKE_CURRENT_SOURCE_DIR}/Platforms/NVIDIA/CUDA-Common/Common/host_options.h" "${CMAKE_CURRENT_SOURCE_DIR}/Include/AWE/awe.h" "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/engines.json" "${_awe_exports}"
        COMMENT "Compiling ${_awe_id} for ${AWE_CUDA_ARCH} with recorded arguments"
        VERBATIM)
    add_custom_target(awe_Inline_${_awe_id} ALL DEPENDS "${_awe_output}")
    add_library(awe::Inline_${_awe_id} SHARED IMPORTED GLOBAL)
    set_target_properties(awe::Inline_${_awe_id} PROPERTIES IMPORTED_LOCATION "${_awe_output}")
    add_dependencies(awe::Inline_${_awe_id} awe_Inline_${_awe_id})
    if(AWE_BUILD_TEMPLATE AND _awe_id MATCHES "^k(64|96)_d(17|18)$")
        string(REPLACE "/CUDA-Inline/" "/CUDA-Template/" _awe_template_path "${_awe_path}")
        set(_awe_template_source "${CMAKE_CURRENT_SOURCE_DIR}/${_awe_template_path}")
        set(_awe_template_output "${CMAKE_CURRENT_BINARY_DIR}/Lib/Template/libawe_${_awe_id}.so")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Lib/Template")
        file(GLOB_RECURSE _awe_template_dependencies CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/Platforms/NVIDIA/CUDA-Template/GEMM/Common/*")
        add_custom_command(OUTPUT "${_awe_template_output}"
            COMMAND "${CMAKE_COMMAND}" -E env --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS --unset=NVCC_CCBIN
                "${AWE_NVCC}" ${_awe_args} -o "${_awe_template_output}"
            WORKING_DIRECTORY "${_awe_template_source}"
            DEPENDS ${_awe_dependencies} ${_awe_template_dependencies}
                "${_awe_template_source}/l8rt.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/Platforms/NVIDIA/CUDA-Common/Common/host_options.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/Include/AWE/awe.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/Docs/Provenance/engines.json" "${_awe_exports}"
            COMMENT "Compiling Template ${_awe_id} for comparison with Inline"
            VERBATIM)
        add_custom_target(awe_Template_${_awe_id} ALL DEPENDS "${_awe_template_output}")
    endif()
endforeach()

get_filename_component(_awe_cuda_bin "${AWE_NVCC}" DIRECTORY)
get_filename_component(_awe_cuda_root "${_awe_cuda_bin}" DIRECTORY)
find_path(AWE_CUDA_INCLUDE cuda_runtime.h HINTS "${_awe_cuda_root}/include" REQUIRED NO_DEFAULT_PATH)
find_library(AWE_CUDART cudart HINTS "${_awe_cuda_root}/lib64" REQUIRED NO_DEFAULT_PATH)
add_library(awe SHARED API/api.cpp)
target_compile_features(awe PRIVATE cxx_std_17)
target_include_directories(awe PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Include" PRIVATE "${AWE_CUDA_INCLUDE}" "${CMAKE_CURRENT_SOURCE_DIR}/Routines/Common")
target_link_libraries(awe PRIVATE awe_engine_loader "${AWE_CUDART}")
target_link_options(awe PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/api_exports.map")
set_target_properties(awe PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Lib"
    BUILD_RPATH "${_awe_cuda_root}/lib64" LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/CMake/api_exports.map")
if(AWE_BACKEND STREQUAL "K96" OR AWE_BACKEND STREQUAL "K64")
    string(REPLACE "sm_" "compute_" _support_compute "${_awe_arch}")
    foreach(_support IN ITEMS real complex float)
        if(_support STREQUAL "real")
            set(_std c++20)
        else()
            set(_std c++17)
        endif()
        set(_support_output "${CMAKE_CURRENT_BINARY_DIR}/Lib/libawe_level3_${_support}.so")
        add_custom_command(OUTPUT "${_support_output}"
            COMMAND "${CMAKE_COMMAND}" -E env --unset=NVCC_PREPEND_FLAGS --unset=NVCC_APPEND_FLAGS --unset=NVCC_CCBIN
                "${AWE_NVCC}" -O3 -std=${_std} -gencode arch=${_support_compute},code=${_awe_arch} --cudart=shared -Xcompiler=-fPIC -shared
                "${CMAKE_CURRENT_SOURCE_DIR}/Platforms/NVIDIA/CUDA-Common/Level3/${_support}.cu"
                -Xlinker=-Bsymbolic "-Xlinker=--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/support_exports.map"
                -Xlinker=-soname=libawe_level3_${_support}.so -o "${_support_output}"
            DEPENDS "Platforms/NVIDIA/CUDA-Common/Level3/${_support}.cu" CMake/support_exports.map VERBATIM)
        add_custom_target(awe_level3_${_support} ALL DEPENDS "${_support_output}")
    endforeach()
endif()

include(CTest)
option(AWE_ENABLE_GPU_TESTS "Register tests that execute CUDA kernels" OFF)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME cpu_contracts COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s Tests)
    set_tests_properties(cpu_contracts PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" LABELS cpu)
    add_test(NAME rational_reference COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/Tools/awe_validation.py")
    set_tests_properties(rational_reference PROPERTIES LABELS cpu)
    if(AWE_ENABLE_GPU_TESTS)
        add_test(NAME gpu_contracts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/Tools/awe_validation.py"
            --gpu --backend ${_awe_k} --library "$<TARGET_FILE:awe>" --cudart "${AWE_CUDART}"
            --output "${CMAKE_CURRENT_BINARY_DIR}/validation-k${_awe_k}.json")
        set_tests_properties(gpu_contracts PROPERTIES LABELS gpu TIMEOUT 1800)
    endif()
endif()
