// exact-dft — L1: naive exact DFT oracle (bit-exact reference).
//
// O(n^2) matrix-vector product in K = Q(zeta_n):  y_k = sum_j x_j * zeta^{jk}.
// This is THE reference every later layer must reproduce bit-exactly
// (no tolerances anywhere in the exact path; spec "Oracle & differential
// testing plan").
#pragma once
#include "cyclotomic.hpp"

#include <stdexcept>
#include <vector>

namespace ed {

// Naive exact DFT of length n over Q(zeta_n). Inputs must live in ring d=n.
inline std::vector<Cyclotomic> oracle_dft(int n, const std::vector<Cyclotomic>& x) {
    if ((int)x.size() != n)
        throw std::invalid_argument("oracle_dft: input length mismatch");
    if (n < 1) throw std::invalid_argument("oracle_dft: n must be >= 1");
    if (n == 1) return x;  // F_1 = [1]
    const Cyclotomic z = Cyclotomic::zeta(n);
    std::vector<Cyclotomic> y;
    y.reserve(n);
    for (int k = 0; k < n; ++k) {
        Cyclotomic acc(n);
        for (int j = 0; j < n; ++j) {
            if (x[j].is_zero()) continue;
            acc += x[j] * z.pow(((int64_t)j * k) % n);
        }
        y.push_back(std::move(acc));
    }
    return y;
}

// Naive exact linear convolution (for Bluestein testing in later milestones).
inline std::vector<Cyclotomic> oracle_conv_linear(const std::vector<Cyclotomic>& a,
                                                  const std::vector<Cyclotomic>& b) {
    if (a.empty() || b.empty() || a[0].d() != b[0].d())
        throw std::invalid_argument("oracle_conv_linear: rings must match");
    const int d = a[0].d();
    std::vector<Cyclotomic> c(a.size() + b.size() - 1, Cyclotomic(d));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] += a[i] * b[j];
    return c;
}

// Embed a rational-complex pair (a + b*i) into Q(zeta_n) via i = zeta_n^{n/4}.
// Requires 4 | n (true for n divisible by 4). Convenience for tests/bindings.
inline Cyclotomic embed_complex(int n, Rational re, Rational im) {
    Cyclotomic e = Cyclotomic::rational(n, re);
    if (!im.is_zero()) {
        if (n % 4 != 0)
            throw std::invalid_argument("embed_complex: need 4 | n to host i in Q(zeta_n)");
        const Cyclotomic z = Cyclotomic::zeta(n);
        e += Cyclotomic::rational(n, im) * z.pow(n / 4);
    }
    return e;
}

}  // namespace ed
