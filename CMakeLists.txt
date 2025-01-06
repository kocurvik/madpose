cmake_minimum_required(VERSION 3.17)
project(madpose)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version needs to be at least 9.1")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++17 -std=gnu++17 -fPIC") # brute-force approach to make sure we are using C++17

find_package(Eigen3 3.4 REQUIRED)
find_package(Ceres 2.0.0 REQUIRED)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(PoseLib REQUIRED)
find_package(OpenCV REQUIRED)
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changes the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()

add_subdirectory(ext/pybind11)

include_directories(BEFORE
    ${pybind11_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ext/RansacLib
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_subdirectory(src)
