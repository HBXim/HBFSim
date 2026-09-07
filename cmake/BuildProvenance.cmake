set(SOURCE_COMMIT "unknown")
set(SOURCE_TREE "unknown")
set(SOURCE_DIRTY 0)
if(GIT_EXECUTABLE)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_ROOT}" rev-parse --verify HEAD
    OUTPUT_VARIABLE COMMIT_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE COMMIT_RESULT
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_ROOT}" rev-parse --verify "HEAD^{tree}"
    OUTPUT_VARIABLE TREE_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE TREE_RESULT
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_ROOT}"
      status --porcelain --untracked-files=normal
    OUTPUT_VARIABLE STATUS_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE STATUS_RESULT
  )
  if(COMMIT_RESULT EQUAL 0 AND TREE_RESULT EQUAL 0 AND STATUS_RESULT EQUAL 0)
    set(SOURCE_COMMIT "${COMMIT_OUTPUT}")
    set(SOURCE_TREE "${TREE_OUTPUT}")
    if(NOT STATUS_OUTPUT STREQUAL "")
      set(SOURCE_DIRTY 1)
    endif()
  endif()
endif()

file(GLOB_RECURSE SOURCE_FILES RELATIVE "${SOURCE_ROOT}"
  "${SOURCE_ROOT}/src/*.cpp" "${SOURCE_ROOT}/src/*.hpp"
  "${SOURCE_ROOT}/cmake/*.cmake"
)
list(APPEND SOURCE_FILES "CMakeLists.txt")
list(SORT SOURCE_FILES)
set(SOURCE_MANIFEST "hbfsim-cxx-source-v1\n")
foreach(SOURCE_FILE IN LISTS SOURCE_FILES)
  file(SHA256 "${SOURCE_ROOT}/${SOURCE_FILE}" FILE_SHA256)
  string(APPEND SOURCE_MANIFEST "${FILE_SHA256}  ${SOURCE_FILE}\n")
endforeach()
string(SHA256 SOURCE_SHA256 "${SOURCE_MANIFEST}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_DIR}/hbfsim_source_manifest.txt.tmp" "${SOURCE_MANIFEST}")
configure_file("${OUTPUT_DIR}/hbfsim_source_manifest.txt.tmp"
  "${OUTPUT_DIR}/hbfsim_source_manifest.txt" COPYONLY)
file(WRITE "${OUTPUT_DIR}/hbfsim_build_provenance.hpp.tmp"
  "#pragma once\n"
  "#define HBFSIM_GIT_COMMIT \"${SOURCE_COMMIT}\"\n"
  "#define HBFSIM_GIT_TREE \"${SOURCE_TREE}\"\n"
  "#define HBFSIM_GIT_DIRTY ${SOURCE_DIRTY}\n"
  "#define HBFSIM_SOURCE_SHA256 \"${SOURCE_SHA256}\"\n"
)
configure_file("${OUTPUT_DIR}/hbfsim_build_provenance.hpp.tmp"
  "${OUTPUT_DIR}/hbfsim_build_provenance.hpp" COPYONLY)
file(REMOVE "${OUTPUT_DIR}/hbfsim_source_manifest.txt.tmp"
  "${OUTPUT_DIR}/hbfsim_build_provenance.hpp.tmp")
