# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pynetcor/)
    endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

# Version info
set(PNC_VERSION_MAJOR "0" CACHE STRING "Major version number")
set(PNC_VERSION_MINOR "0" CACHE STRING "Minor version number")
set(PNC_VERSION_PATCH "0" CACHE STRING "Patch version number")

add_subdirectory(../external/pybind11 pybind11_build)
add_subdirectory(../src pyNetCor_build)

set(SOURCES _core/pyNetCor.cpp _core/cor.cpp _core/chunked_cor.cpp _core/cluster.cpp)

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/_core)
pybind11_add_module(_core ${SOURCES})
target_link_libraries(_core PRIVATE netcor)
target_compile_definitions(_core PRIVATE
        PNC_VERSION_MAJOR=${PNC_VERSION_MAJOR}
        PNC_VERSION_MINOR=${PNC_VERSION_MINOR}
        PNC_VERSION_PATCH=${PNC_VERSION_PATCH}
)
set_target_properties(_core PROPERTIES PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")