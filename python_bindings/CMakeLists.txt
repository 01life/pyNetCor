add_subdirectory(../external/pybind11 pybind11_build)
add_subdirectory(../pyNetCor pyNetCor_build)

set(SOURCES _core/pyNetCor.cpp _core/cor.cpp _core/chunked_cor.cpp _core/cluster.cpp)

pybind11_add_module(_core ${SOURCES})
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/_core)
target_link_libraries(_core PRIVATE pyNetCor)