cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)

project(FionaPhotonic_Spike)
project(FionaPhotonic_Verilator)

# Build options
option(USE_PYTHON "Use Python photonic models (default). Set to OFF for pure C++ mode." ON)
option(USE_EIGEN "Enable Eigen3 library for matrix operations" OFF)

set(CMAKE_C_STANDARD 11)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# C++ mode requires Eigen
if (NOT USE_PYTHON AND NOT USE_EIGEN)
  message(STATUS "Pure C++ mode requires Eigen. Enabling USE_EIGEN automatically.")
  set(USE_EIGEN ON)
endif()

if (USE_EIGEN)
  add_definitions(-DUSE_EIGEN)
  find_package(Eigen3 3.3 REQUIRED)
  message(STATUS "Eigen3 is enabled for Spike-Simulator.")
endif ()

# Python is only required when USE_PYTHON is ON
if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python 3.6 REQUIRED Development)
  message(STATUS "Python photonic model mode (default)")
else()
  message(STATUS "")
  message(STATUS "========================================")
  message(STATUS "  Pure C++ Photonic Model Mode")
  message(STATUS "  - No Python dependency")
  message(STATUS "  - ~50x faster than Python mode")
  message(STATUS "========================================")
  message(STATUS "")
endif()

find_package(verilator HINTS $ENV{VERILATOR_ROOT})
find_package(Threads REQUIRED)

if (NOT verilator_FOUND)
  message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable.")
endif()

if (USE_EIGEN)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif ()
if (USE_PYTHON)
  include_directories(${Python_INCLUDE_DIRS})
endif()
include_directories(${PROJECT_SOURCE_DIR} bridge)

# spike
add_executable(FionaPhotonic_Spike test/spike/main.cpp)
target_include_directories(FionaPhotonic_Spike PRIVATE bridge/spike)
if (USE_PYTHON)
  target_link_libraries(FionaPhotonic_Spike PRIVATE ${Python_LIBRARIES} ${CMAKE_DL_LIBS})
else()
  # C++ mode: no Python libs, just dl for potential future plugins
  target_link_libraries(FionaPhotonic_Spike PRIVATE ${CMAKE_DL_LIBS})
endif()

# verilator
add_executable(FionaPhotonic_Verilator bridge/verilator/engine.cc test/verilator/main.cpp)
target_include_directories(FionaPhotonic_Verilator PRIVATE bridge/verilator)
target_link_libraries(FionaPhotonic_Verilator PRIVATE ${Python_LIBRARIES})
target_link_libraries(FionaPhotonic_Verilator PRIVATE Threads::Threads)
set(VERILATOR_TOP_MODULE)
verilate(${PROJECT_NAME}
  INCLUDE_DIRS "bridge/verilator/"
  SOURCES test/verilator/top.sv
)

file(COPY "${PROJECT_SOURCE_DIR}/pyfunc/" DESTINATION "${PROJECT_BINARY_DIR}/pyfunc/")

