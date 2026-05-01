cmake_minimum_required(VERSION 3.20)

project(HBFSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
