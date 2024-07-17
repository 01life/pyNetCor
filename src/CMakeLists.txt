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
if(WIN32)
    find_library(OpenBLAS_LIB NAMES libopenblas openblas
            HINTS ${OpenBLAS_INCLUDE_DIRS}/../../lib ${OpenBLAS_INCLUDE_DIRS}/../../lib64)
elseif(APPLE)
    find_library(OpenBLAS_LIB NAMES libopenblas openblas
            HINTS ${OpenBLAS_INCLUDE_DIRS}/../../lib ${OpenBLAS_INCLUDE_DIRS}/../../lib64)
elseif(UNIX)
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

add_library(netcor STATIC ${SOURCES})

# Set properties for netcor
set_target_properties(netcor PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        MACOSX_RPATH ON
        POSITION_INDEPENDENT_CODE ON
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
# Include directories and link libraries
target_include_directories(netcor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/external
        ${OpenBLAS_INCLUDE_DIRS}
)
message(STATUS "Link OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
target_link_libraries(netcor ${OpenMP_CXX_LIBRARIES} ${OpenBLAS_LIBRARIES})