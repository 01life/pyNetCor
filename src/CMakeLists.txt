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

set(SOURCES cor.cpp padjusttable.cpp ptable.cpp util.cpp cluster.cpp)

add_library(netcor STATIC ${SOURCES})
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