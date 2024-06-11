#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_cor(py::module &m);
void bind_chunked_cor(py::module &m);
void bind_cluster(py::module &m);

PYBIND11_MODULE(_core, m) {
    bind_cor(m);
    bind_chunked_cor(m);
    bind_cluster(m);
}
