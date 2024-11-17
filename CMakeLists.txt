cmake_minimum_required(VERSION 3.15)

project(pynetcor)

set(CMAKE_CXX_STANDARD 17)
message(STATUS "CMAKE_CXX_STANDARD 17")

# Windows ARM64
if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
    option(PYBIND11_PYTHONLIBS_OVERWRITE "" OFF)

    set(PYTHON_PREFIX "$ENV{LOCALAPPDATA}/pypa/cibuildwheel/Cache/nuget-cpython/pythonarm64.$ENV{PYTHON_VERSION}/tools")
    if(NOT DEFINED $ENV{CIBUILDWHEEL})
        message(WARNING
                " This is hack for cibuildwheel on github action\n"
                " Use the right way to cross-compile python module for windows arm64 like follows\n"
                " set(PYTHON_PREFIX \"<your-pythonarm64-root-path>\")\n"
        )
    endif()
endif()

add_subdirectory(pynetcor)