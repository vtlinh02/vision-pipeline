# Purpose: Build Week One demo (C++17) linking OpenCV + ONNX Runtime, plus GoogleTest unit tests.

cmake_minimum_required(VERSION 3.20)
project(visioncore_week_one LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(VC_BUILD_TESTS "Build unit tests" ON)

# Defaults for performance builds.
# User requested recommended flags: -O3 -march=native -DNDEBUG
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -DNDEBUG)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(ONNXRuntime REQUIRED)

add_library(visioncore
  src/config.cpp
  src/metrics.cpp
  src/preprocess.cpp
  src/ort_session.cpp
  src/postprocess.cpp
  src/json_writer.cpp
  src/overlay.cpp
)
target_include_directories(visioncore PUBLIC include)
target_link_libraries(visioncore PUBLIC ${OpenCV_LIBS} ONNXRuntime::onnxruntime)

add_executable(detect_image apps/detect_image/main.cpp)
target_link_libraries(detect_image PRIVATE visioncore)

set_target_properties(detect_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")