# OpenBLAS
if(WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(OpenBLAS REQUIRED openblas)
    find_library(OpenBLAS_LIB NAMES libopenblas openblas
            HINTS ${OpenBLAS_INCLUDE_DIRS}/../../lib ${OpenBLAS_INCLUDE_DIRS}/../../lib64)
elseif(APPLE)
    find_package(OpenBLAS REQUIRED)
    find_library(OpenBLAS_LIB NAMES libopenblas openblas
            HINTS ${OpenBLAS_INCLUDE_DIRS}/../../lib ${OpenBLAS_INCLUDE_DIRS}/../../lib64)
elseif(UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(OpenBLAS REQUIRED openblas)
    find_library(OpenBLAS_LIB NAMES libopenblas openblas
            HINTS ${OpenBLAS_INCLUDE_DIRS}/../../lib ${OpenBLAS_INCLUDE_DIRS}/../../lib64)
endif()
if(OpenBLAS_LIB)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIB})
    message(STATUS "Found OpenBLAS library: ${OpenBLAS_LIB}")
else()
    message(FATAL_ERROR "OpenBLAS library not found in ${OpenBLAS_INCLUDE_DIRS}/../../lib")
endif()
message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")

set(SOURCES cor.cpp padjusttable.cpp ptable.cpp util.cpp cluster.cpp)

add_library(netcor SHARED ${SOURCES})

# Set properties for netcor
set_target_properties(netcor PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        MACOSX_RPATH ON
        POSITION_INDEPENDENT_CODE ON
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build"
)
# Include directories and link libraries
target_include_directories(netcor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/external
        ${OpenBLAS_INCLUDE_DIRS}
)
message(STATUS "Link OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
target_link_libraries(netcor ${OpenMP_CXX_LIBRARIES} ${OpenBLAS_LIBRARIES})