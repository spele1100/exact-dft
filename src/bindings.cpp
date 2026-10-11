// exact-dft — L6: pybind11 surface (spec L6).
//
//   zeta(d)                       -> Cyclotomic   (the class of t in Q(zeta_d))
//   Cyclotomic                    exact ring element: +, -, *, pow, ==,
//                                 coeffs() (exact rationals), as_complex()
//                                 (only for d in {1, 2, 4}: Gaussian rationals)
//   exact_dft(x)                  exact F_n x for a list of complex (each
//                                 double component converted EXACTLY to its
//                                 dyadic rational value) or Cyclotomic
//   oracle_dft(n, x)              naive exact reference
//   exact_convolve(a, b)          exact linear convolution (Corollary 1.3)
//   working_length(n)             L in [2n, 4n), odd-prime prefix + 2-power
//   compile_report(n)             structure/op-count introspection
//   network_report(h)             scaled sec-2 dimension budget incl. lambda(h)
//   network_exchange_demo(h)      runs the scalar bank exchange with garbage
//                                 aux and verifies (X, Y) -> (-Y, X) exactly
//
// Honest positioning: everything is exact over cyclotomic fields; no float
// path, no stability claims (see README).
#include "exact/cyclotomic.hpp"
#include "exact/dft.hpp"
#include "exact/lengths.hpp"
#include "exact/network.hpp"
#include "exact/oracle_dft.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <cmath>
#include <random>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;
using ed::Cyclotomic;
using ed::Rational;

// Exact dyadic conversion: every finite double is p * 2^e with |p| < 2^53,
// so the conversion is lossless for all normal magnitudes.
static Rational rational_from_double(double v) {
    if (!std::isfinite(v))
        throw std::invalid_argument("exact_dft: non-finite input");
    if (v == 0.0) return Rational(0);
    int exp;
    const double fr = std::frexp(v, &exp);  // v = fr * 2^exp, fr in [0.5, 1)
    const int64_t num = (int64_t)std::ldexp(fr, 53);  // exact
    const int e = exp - 53;                 // v = num * 2^e
    if (e > 62) throw std::overflow_error("exact_dft: magnitude too large for exact int64 conversion");
    if (e < -62) throw std::overflow_error("exact_dft: magnitude too small (subnormal) for exact conversion");
    if (e >= 0) return Rational(num * (int64_t(1) << e));
    return Rational(num, int64_t(1) << -e);
}

// Gaussian rational a + bi embedded into Q(zeta_D) (requires 4 | D).
static Cyclotomic gauss_in_ring(int D, const Rational& re, const Rational& im) {
    if (D % 4 != 0)
        throw std::logic_error("gauss_in_ring: internal error (4 | D required)");
    Cyclotomic c = Cyclotomic::rational(D, re);
    if (!im.is_zero())
        c += Cyclotomic::rational(D, im) * Cyclotomic::zeta(D).pow(D / 4);
    return c;
}


// Exact Python int from __int128 (via decimal string).
static py::object pybig(ed::net::i128 v) {
    return py::module_::import("builtins").attr("int")(ed::net::i128_str(v));
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "exact-dft C++ core: exact DFT over cyclotomic fields "
              "(component-level implementation of OpenAI's explicit "
              "power-saving exact DFT paper)";
    m.attr("__version__") = "0.1.0";
    m.attr("gmp_backend") = ed::kGMPBackend;

    py::class_<Cyclotomic>(m, "Cyclotomic")
        .def(py::init<int>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(-py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("pow", &Cyclotomic::pow, py::arg("k"))
        .def("is_zero", &Cyclotomic::is_zero)
        .def_property_readonly("d", &Cyclotomic::d)
        .def_property_readonly("degree", &Cyclotomic::degree)
        .def("coeffs",
             [](const Cyclotomic& c) {
                 py::list out;
                 for (const auto& q : c.coeffs())
                     out.append(py::make_tuple(q.num_i64(), q.den_i64()));
                 return out;
             })
        .def("as_complex",
             [](const Cyclotomic& c) {
                 const int d = c.d();
                 // Exact Gaussian-rational extraction: the value lies in Q(i)
                 // (4 | d) iff it is fixed by every Galois automorphism
                 // sigma_k with gcd(k, d) = 1, k = 1 (mod 4) — those are
                 // exactly the automorphisms fixing i = zeta_d^{d/4}. Then
                 // a = (c + conj c)/2, b = (c - conj c)/(2i) are RATIONALS,
                 // so their reduced coefficient vectors are [a, 0, ...].
                 if (d == 1 || d == 2) {
                     return std::complex<double>(c.coeffs()[0].to_double(), 0.0);
                 }
                 if (d % 4 != 0)
                     throw std::invalid_argument(
                         "as_complex: value does not lie in Q(i) (ring Q(zeta_d) "
                         "with 4 not dividing d)");
                 const Cyclotomic z = Cyclotomic::zeta(d);
                 auto gal = [&](int64_t k) {
                     const Cyclotomic zk = z.pow(k);
                     Cyclotomic acc(d);
                     Cyclotomic wp = Cyclotomic::integer(d, 1);
                     const auto& co = c.coeffs();
                     for (size_t j = 0; j < co.size(); ++j) {
                         if (!co[j].is_zero()) acc += Cyclotomic::rational(d, co[j]) * wp;
                         if (j + 1 < co.size()) wp = wp * zk;
                     }
                     return acc;
                 };
                 for (int64_t k = 1; k < d; ++k) {
                     if (k % 4 != 1 || std::gcd((int64_t)k, (int64_t)d) != 1) continue;
                     if (!(gal(k) == c))
                         throw std::invalid_argument(
                             "as_complex: value is an exact element of Q(zeta_d) "
                             "that does NOT lie in Q(i)");
                 }
                 const Cyclotomic conj = gal(d - 1);
                 const Rational half(1, 2);
                 const Cyclotomic re = (c + conj) * Cyclotomic::rational(d, half);
                 const Cyclotomic im = (c - conj) *
                     (Cyclotomic::rational(d, Rational(-1, 2)) * z.pow(d / 4));
                 return std::complex<double>(re.coeffs()[0].to_double(),
                                             im.coeffs()[0].to_double());
             },
             "exact float view: only when the VALUE lies in Q(i) (tested exactly "
             "via Galois invariance); the returned pair is the exact Gaussian "
             "rational (a, b) rendered as doubles")
        .def("__repr__", [](const Cyclotomic& c) {
            std::string s = "Cyclotomic(d=" + std::to_string(c.d()) + ", coeffs=[";
            for (size_t i = 0; i < c.coeffs().size(); ++i) {
                if (i) s += ", ";
                s += c.coeffs()[i].str();
            }
            return s + "])";
        });

    m.def("zeta", &Cyclotomic::zeta, py::arg("d"),
          "the class of t (a primitive d-th root of unity) in Q(zeta_d)");
    m.def(
        "gauss",
        [](int d, double re, double im) {
            return gauss_in_ring(d, rational_from_double(re),
                                 rational_from_double(im));
        },
        py::arg("d"), py::arg("re") = 0.0, py::arg("im") = 0.0,
        "exact Gaussian rational a + bi in Q(zeta_d) (requires 4 | d);\n"
        "the double components are converted EXACTLY (dyadic rationals)");
    m.def("oracle_dft", &ed::oracle_dft, py::arg("n"), py::arg("x"),
          "naive exact O(n^2) DFT reference (inputs in ring n)");

    // exact_dft: complex list (exact dyadic conversion) or Cyclotomic list.
    m.def(
        "exact_dft",
        [](const py::list& x) {
            const size_t n = x.size();
            if (n == 0) throw std::invalid_argument("exact_dft: empty input");
            std::vector<std::optional<std::complex<double>>> cz(n);
            int boost = 1;
            for (size_t i = 0; i < n; ++i) {
                const py::handle h = x[i];
                if (PyComplex_Check(h.ptr()) || PyFloat_Check(h.ptr()) ||
                    PyLong_Check(h.ptr())) {
                    if (PyComplex_Check(h.ptr())) {
                        cz[i] = h.cast<std::complex<double>>();
                    } else {
                        cz[i] = std::complex<double>(h.cast<double>(), 0.0);
                    }
                    boost = std::lcm(boost, 4);  // host i
                } else {
                    const Cyclotomic& c = h.cast<const Cyclotomic&>();
                    boost = std::lcm(boost, c.d());
                }
            }
            ed::ExactDFT drv((int)n, boost);
            std::vector<Cyclotomic> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (cz[i]) {
                    v.push_back(gauss_in_ring(drv.ring(),
                                              rational_from_double(cz[i]->real()),
                                              rational_from_double(cz[i]->imag())));
                } else {
                    v.push_back(x[i].cast<const Cyclotomic&>());
                }
            }
            return drv.dft(v);
        },
        py::arg("x"),
        "exact F_n x (bit-exact, cyclotomic arithmetic); complex inputs are\n"
        "converted exactly (every finite double is a dyadic rational)");

    m.def(
        "exact_convolve",
        [](const std::vector<Cyclotomic>& a, const std::vector<Cyclotomic>& b) {
            return ed::exact_convolve(a, b);
        },
        py::arg("a"), py::arg("b"),
        "exact linear convolution via three length-L transforms");

    m.def(
        "working_length",
        [](int n) {
            const ed::WorkingLength w = ed::working_length(n);
            py::dict d;
            d["n"] = n;
            d["odd_primes"] = w.odd_primes;
            d["R_odd"] = w.R_odd;
            d["two_exp"] = w.two_exp;
            d["L"] = w.L;
            return d;
        },
        py::arg("n"),
        "working length L = R_odd * 2^e in [2n, 4n) (paper sec 5)");

    m.def(
        "compile_report",
        [](int n) {
            ed::ExactDFT drv(n);
            const ed::DriverReport r = drv.report();
            py::dict d;
            d["n"] = r.n;
            d["L"] = r.L;
            d["ring"] = r.ring;
            d["phi_ring"] = (long long)ed::detail::euler_phi(r.ring);
            d["two_exp"] = r.two_exp;
            d["odd_primes"] = r.odd_primes;
            d["transform_calls"] = r.transform_calls;
            d["radix2_mults"] = r.radix2_mults;
            d["axes_R"] = r.axes.R;
            d["axes_groups"] = r.axes.groups;
            d["axes_prefix_extensions"] = r.axes.prefix_extensions;
            d["axes"] = [r]() {
                py::list out;
                for (const auto& a : r.axes.axes) {
                    py::dict ad;
                    ad["r"] = a.r;
                    ad["shears"] = a.shears;
                    ad["gate_coords"] = a.gate_coords;
                    ad["replays"] = a.replays;
                    ad["max_num_bits"] = a.max_num_bits;
                    ad["max_den_bits"] = a.max_den_bits;
                    out.append(ad);
                }
                return out;
            }();
            return d;
        },
        py::arg("n"),
        "structure/op-count introspection for the compiled length-n pipeline");

    m.def(
        "network_report",
        [](int h) {
            const ed::net::NetworkReport r = ed::net::network_report(h);
            py::dict d;
            d["h"] = r.h;
            d["v"] = ed::net::i128_str(r.v);
            d["m"] = ed::net::i128_str(r.m);
            d["N0"] = ed::net::i128_str(r.N0);
            d["I0"] = ed::net::i128_str(r.I0);
            d["d_nbr"] = ed::net::i128_str(r.d_nbr);
            d["W"] = ed::net::i128_str(r.W);
            d["s"] = ed::net::i128_str(r.s);
            d["Delta"] = ed::net::i128_str(r.Delta);
            d["Wstar"] = ed::net::i128_str(r.Wstar);
            d["S"] = ed::net::i128_str(r.S);
            d["saving"] = r.saving;
            d["lambda"] = (double)r.lambda;
            d["theta"] = (double)r.theta;
            d["lines"] = r.lines();
            d["verdict"] = r.saving
                ? "Delta > 0 but galactic (arrays of 2^m entries); no useful crossover"
                : "NO power saving at this h (lambda > m, theta > 1)";
            return d;
        },
        py::arg("h") = 4,
        "scaled sec-2 tensor-network dimension budget: wire counts, lambda(h),\n"
        "theta(h) and the honest verdict (default h = 4: no saving)");

    m.def(
        "network_exchange_demo",
        [](int h) {
            const ed::net::ScalarNetwork net(h);
            const int v = net.v();
            std::mt19937 rng(12345);
            auto rand_gauss = [&rng]() {
                std::uniform_int_distribution<int64_t> dist(-9, 9);
                return Cyclotomic::rational(4, Rational(dist(rng), 7)) +
                       Cyclotomic::rational(4, Rational(dist(rng), 5)) *
                           Cyclotomic::zeta(4);
            };
            std::vector<Cyclotomic> X, Y;
            for (int i = 0; i < v * v * v; ++i) X.push_back(rand_gauss());
            for (int i = 0; i < v * v * v; ++i) Y.push_back(rand_gauss());
            const auto X0 = X, Y0 = Y;
            auto garbage = [&](std::vector<Cyclotomic>& A, std::vector<Cyclotomic>& c) {
                for (auto& a : A) a = rand_gauss();  // arbitrary aux garbage:
                for (auto& q : c) q = rand_gauss();  // the paper's Lemma net:scalar
            };
            net.exchange(X, Y, garbage);
            bool ok = true;
            for (int i = 0; i < v * v * v; ++i)
                ok = ok && X[i] == -Y0[i] && Y[i] == X0[i];
            py::dict d;
            d["h"] = h;
            d["v"] = v;
            d["bank_entries"] = v * v * v;
            d["invocations"] = 3LL * v * v;
            d["verified_exact"] = ok;
            return d;
        },
        py::arg("h") = 4,
        "run the scaled scalar bank exchange (8-row schedule, 3 stages) with\n"
        "arbitrary garbage in every auxiliary wire; verifies (X, Y) -> (-Y, X)");
}
