cmake_minimum_required(VERSION 3.20)

project(HBFSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Git QUIET)
add_custom_target(hbfsim_build_provenance
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${CMAKE_SOURCE_DIR}"
    "-DOUTPUT_DIR=${CMAKE_BINARY_DIR}/generated"
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    -P "${CMAKE_SOURCE_DIR}/cmake/BuildProvenance.cmake"
  BYPRODUCTS
    "${CMAKE_BINARY_DIR}/generated/hbfsim_build_provenance.hpp"
    "${CMAKE_BINARY_DIR}/generated/hbfsim_source_manifest.txt"
  VERBATIM
)

add_library(hbfsim_core
  src/physical/address_heatmap.cpp
  src/physical/base_die_link.cpp
  src/physical/external/cxl_ssd.cpp
  src/physical/external/external_backing_device.cpp
  src/physical/hbf/hbf_device.cpp
  src/physical/hbf/hbf_persistent_image.cpp
  src/physical/hbm/hbm_device.cpp
  src/physical/simulation_session.cpp
)

add_library(HBFSim::core ALIAS hbfsim_core)

target_include_directories(hbfsim_core PUBLIC src)

target_compile_options(hbfsim_core PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

add_library(hbfsim_engine_app
  src/app/session_protocol.cpp
  src/app/system_config.cpp
)

target_include_directories(hbfsim_engine_app PUBLIC
  src
  "${CMAKE_BINARY_DIR}/generated"
)
add_dependencies(hbfsim_engine_app hbfsim_build_provenance)
target_link_libraries(hbfsim_engine_app PUBLIC hbfsim_core)
target_compile_options(hbfsim_engine_app PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

add_executable(hbfsim src/app/session_main.cpp)
target_link_libraries(hbfsim PRIVATE hbfsim_engine_app)
target_compile_definitions(hbfsim PRIVATE
  HBFSIM_VERSION="${PROJECT_VERSION}"
  HBFSIM_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
  HBFSIM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  HBFSIM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)
