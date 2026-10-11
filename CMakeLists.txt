cmake_minimum_required(VERSION 3.21)
project(exact-dft
    VERSION 0.1.0
    DESCRIPTION "Exact DFT: component-level reference implementation of OpenAI's explicit power-saving exact DFT paper"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EXACT_DFT_BUILD_TESTS "Build test suites" ON)
option(EXACT_DFT_BUILD_PYTHON "Build pybind11 Python module" OFF)
option(EXACT_DFT_USE_GMP "Use GMP-backed big rationals when available" OFF)

add_library(exact_dft INTERFACE)
target_include_directories(exact_dft INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
target_compile_features(exact_dft INTERFACE cxx_std_20)

if(EXACT_DFT_USE_GMP)
    find_path(GMP_INCLUDE_DIR gmpxx.h)
    find_library(GMPXX_LIB NAMES gmpxx)
    find_library(GMP_LIB NAMES gmp)
    if(GMP_INCLUDE_DIR AND GMPXX_LIB AND GMP_LIB)
        message(STATUS "exact-dft: GMP backend ON (${GMPXX_LIB})")
        target_include_directories(exact_dft INTERFACE ${GMP_INCLUDE_DIR})
        target_link_libraries(exact_dft INTERFACE ${GMPXX_LIB} ${GMP_LIB})
        target_compile_definitions(exact_dft INTERFACE ED_HAS_GMP)
    else()
        message(WARNING "exact-dft: EXACT_DFT_USE_GMP=ON but GMP not found; falling back to int64 backend.")
    endif()
endif()


if(EXACT_DFT_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(WARNING "pybind11 not found; Python module skipped (pip install pybind11).")
    else()
        pybind11_add_module(_core MODULE src/bindings.cpp)
        target_link_libraries(_core PRIVATE exact_dft)
        install(TARGETS _core DESTINATION exact_dft)
        install(FILES src/python/exact_dft/__init__.py DESTINATION exact_dft)
    endif()
endif()

if(EXACT_DFT_BUILD_TESTS)
    enable_testing()
    foreach(t test_rational test_cyclotomic test_oracle test_newton_toeplitz test_schedule test_driver test_network)
        add_executable(${t} tests/${t}.cpp)
        target_link_libraries(${t} PRIVATE exact_dft)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()
