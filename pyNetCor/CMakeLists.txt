set(SOURCES cor.cpp padjusttable.cpp ptable.cpp util.cpp cluster.cpp txtreader.cpp)

add_library(pyNetCor SHARED ${SOURCES})

target_include_directories(pyNetCor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pyNetCor ${OpenMP_CXX_LIBRARIES} ${OpenBLAS_LIBRARIES} ${ZLIB_LIBRARIES} )