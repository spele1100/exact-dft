// exact-dft — L6 (M1 skeleton): pybind11 module wiring.
// Surface grows in later milestones (exact_dft, exact_convolve, zeta,
// compile_report per spec L6). For M1 it exposes the exact ring so the
// packaging path can be smoke-tested early.
#include "exact/cyclotomic.hpp"
#include "exact/oracle_dft.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using ed::Cyclotomic;

PYBIND11_MODULE(_core, m) {
    m.doc() = "exact-dft C++ core (M1: exact ring + oracle only)";
    m.def("zeta", [](int d) { return Cyclotomic::zeta(d); });
    m.def("oracle_dft",
          [](int n, const std::vector<Cyclotomic>& x) { return ed::oracle_dft(n, x); });
    py::class_<Cyclotomic>(m, "Cyclotomic")
        .def(py::init<>())
        .def("__repr__", [](const Cyclotomic& c) {
            std::string s = "Cyclotomic(d=" + std::to_string(c.d()) + ", coeffs=[";
            for (size_t i = 0; i < c.coeffs().size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(c.coeffs()[i].num) + "/" + std::to_string(c.coeffs()[i].den);
            }
            return s + "])";
        });
}
