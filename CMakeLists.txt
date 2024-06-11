cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

project(pyNetCor)


# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


# OpenBLAS
find_package(OpenBLAS REQUIRED)
if (OpenBLAS_FOUND)
    message(STATUS "OpenBLAS found")
    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
endif ()

# blaze
#set(BLAZE_INCLUDE_DIR /aimigene/longshibin/cpplib/blaze-3.8.2/include)

include_directories(${OpenBLAS_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})

add_subdirectory(python_bindings)