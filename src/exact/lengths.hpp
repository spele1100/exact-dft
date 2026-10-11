// exact-dft — L5a: working-length selection (paper sec 5, all-lengths.tex).
//
// The paper picks ell = Theta(log n / loglog n) initial odd primes (no PNT
// needed, elementary bounds), takes their product R_ell <= 2n, and pads with
// one power of two to a working length L = R_ell * 2^e in [2n, 4n):
//   * largest ell with R_ell = prod of the first ell odd primes (3, 5, 7, ...)
//     satisfying R_ell <= 2n;
//   * e minimal with R_ell * 2^e >= 2n. Then L < 4n because L/2 < 2n.
//
// The radices (the ell odd primes + e radix-2 stages) are pairwise coprime,
// which is what the scheduled tensor transform (L3) and the Good–Thomas
// coprime-axis assembly in dft.hpp rely on.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ed {

// Trial-division primality (paper needs no PNT; neither do we).
inline bool is_prime_ti(int64_t v) {
    if (v < 2) return false;
    for (int64_t p = 2; p * p <= v; ++p)
        if (v % p == 0) return false;
    return true;
}

struct WorkingLength {
    int n = 0;
    std::vector<int> odd_primes;  // 3, 5, 7, ... prefix
    long long R_odd = 1;          // product of odd_primes
    int two_exp = 0;              // power-of-two padding
    int L = 0;                    // R_odd << two_exp, in [2n, 4n)
};

inline WorkingLength working_length(int n) {
    if (n < 1) throw std::invalid_argument("working_length: n must be >= 1");
    WorkingLength w;
    w.n = n;
    const int64_t bound = 2LL * n;
    long long R = 1;
    for (int64_t p = 3;; p += 2) {
        if (!is_prime_ti(p)) continue;
        if (R * p > bound) break;
        w.odd_primes.push_back((int)p);
        R *= p;
    }
    w.R_odd = R;
    int e = 0;
    while ((R << e) < bound) ++e;
    w.two_exp = e;
    w.L = (int)(R << e);
    if (!(2LL * n <= w.L && w.L < 4LL * n))
        throw std::logic_error("working_length: invariant 2n <= L < 4n violated");
    return w;
}

}  // namespace ed
