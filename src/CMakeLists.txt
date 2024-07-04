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
    # If on Windows and OpenBLAS_LIBRARIES is an empty string
    if(WIN32 AND "${OpenBLAS_LIBRARIES}" STREQUAL "")
        set(OpenBLAS_LIBRARIES "${OpenBLAS_INCLUDE_DIRS}/../../lib/openblas.lib")
        message(WARNING "OpenBLAS_LIBRARIES was empty. Set to default: ${OpenBLAS_LIBRARIES}")
    endif()
    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
endif ()
#find_package(OpenBLAS)
#if (OpenBLAS_FOUND)
#    # If on Windows and OpenBLAS_LIBRARIES is an empty string
#    if(WIN32 AND "${OpenBLAS_LIBRARIES}" STREQUAL "")
#        set(OpenBLAS_LIBRARIES "${OpenBLAS_INCLUDE_DIRS}/../../lib/openblas.lib")
#        message(WARNING "OpenBLAS_LIBRARIES was empty. Set to default: ${OpenBLAS_LIBRARIES}")
#    endif()
#
#    message(STATUS "OpenBLAS found")
#    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
#    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
#else ()
#    set(OPENBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/OpenBLAS")
#    set(OPENBLAS_INSTALL_DIR "${CMAKE_BINARY_DIR}/openblas-install")
#    message(STATUS "OpenBLAS not found, installing to ${OPENBLAS_INSTALL_DIR}")
#    # Install OpenBLAS
#    include(ExternalProject)
#    ExternalProject_Add(OpenBLAS
#        SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
#        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL_DIR}
#                   -DBUILD_SHARED_LIBS=ON
#                   -DCMAKE_BUILD_TYPE=Release
#        BUILD_COMMAND cmake --build <BINARY_DIR> --config Release
#        INSTALL_COMMAND cmake --install <BINARY_DIR>
#    )
#    set(OpenBLAS_INCLUDE_DIRS ${OPENBLAS_INSTALL_DIR}/include/openblas)
#    if (WIN32)
#        set(OpenBLAS_LIBRARIES ${OPENBLAS_INSTALL_DIR}/lib/openblas.lib)
#    elseif (APPLE)
#        set(OpenBLAS_LIBRARIES ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.dylib)
#    elseif (UNIX)
#        set(OpenBLAS_LIBRARIES ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.so)
#    else()
#        message(FATAL_ERROR "Unsupported platform")
#    endif()
#    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
#    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
#endif ()

set(SOURCES cor.cpp padjusttable.cpp ptable.cpp util.cpp cluster.cpp)

add_library(netcor SHARED ${SOURCES})
set_target_properties(netcor PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        MACOSX_RPATH ON
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
target_include_directories(netcor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_SOURCE_DIR}/external ${OpenBLAS_INCLUDE_DIRS})
message(STATUS "Link OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
target_link_libraries(netcor ${OpenMP_CXX_LIBRARIES} ${OpenBLAS_LIBRARIES})