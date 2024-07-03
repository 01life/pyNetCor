#include <string>
#include <sstream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// PyNetCor version
#ifndef PNC_VERSION_MAJOR
#define PNC_VERSION_MAJOR 0
#endif

#ifndef PNC_VERSION_MINOR
#define PNC_VERSION_MINOR 0
#endif

#ifndef PNC_VERSION_PATCH
#define PNC_VERSION_PATCH 0
#endif

std::string version() {
    std::ostringstream version_stream;
    version_stream << PNC_VERSION_MAJOR << "."
                   << PNC_VERSION_MINOR << "."
                   << PNC_VERSION_PATCH;
    return version_stream.str();
}

void bind_cor(py::module &m);
void bind_chunked_cor(py::module &m);
void bind_cluster(py::module &m);

PYBIND11_MODULE(_core, m) {
m.def("version", &version, "A function that returns the version");
m.attr("__version__") = version();
bind_cor(m);
bind_chunked_cor(m);
bind_cluster(m);
}
