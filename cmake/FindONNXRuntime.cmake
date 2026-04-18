# Purpose: Minimal finder for ONNX Runtime C/C++ prebuilt package (include/ + lib/).
# Expected layout:
#   third_party/onnxruntime/include/onnxruntime_cxx_api.h
#   third_party/onnxruntime/lib/libonnxruntime.so

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  REQUIRED
)

add_library(ONNXRuntime::onnxruntime SHARED IMPORTED)
set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)