// exact-dft — L0: exact rational arithmetic, dual backend.
//
// Primary backend (default, dependency-free): int64 numerator/denominator,
// always gcd-reduced, denominator > 0, all intermediates through __int128
// with explicit overflow detection (throws std::overflow_error).
//
// GMP backend (CMake -DEXACT_DFT_USE_GMP=ON, macro ED_HAS_GMP): arbitrary
// precision mpz_class coefficients, same reduced-fraction semantics, no
// overflow. Rationale (M4): the Bluestein chirp path multiplies coefficient
// magnitudes through length-L convolutions in the composite ring
// Q(zeta_lcm(2n, L)); phi of that ring reaches ~1e3 at n ~ 100 and int64
// overflows well before the spec's fuzz matrix sizes.
//
// Differential validation across backends: every test suite runs in BOTH
// build configurations against the SAME bit-exact oracle with deterministic
// inputs, plus a backend-independent literal check (test_driver, n = 4) —
// so agreement of the two backends on the overlap range is enforced by
// construction, not by running both in one process.
#pragma once
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef ED_HAS_GMP
#include <gmpxx.h>
#endif

namespace ed {

#ifdef ED_HAS_GMP
using CInt = mpz_class;
#else
using CInt = int64_t;
#endif

inline constexpr bool kGMPBackend =
#ifdef ED_HAS_GMP
    true;
#else
    false;
#endif

struct Rational {
    CInt num = 0;
    CInt den = 1;

    Rational() = default;
    Rational(int64_t n) : num(n), den(1) {}
    Rational(int64_t n, int64_t d) : num(n), den(d) { normalize(); }
#ifdef ED_HAS_GMP
    Rational(CInt n, CInt d) : num(std::move(n)), den(std::move(d)) { normalize(); }
#endif

    void normalize() {
        if (den == 0) throw std::invalid_argument("ed::Rational: zero denominator");
#ifdef ED_HAS_GMP
        if (sgn(den) < 0) { num = -num; den = -den; }
        CInt g;
        mpz_gcd(g.get_mpz_t(), num.get_mpz_t(), den.get_mpz_t());
        if (g > 1) { num /= g; den /= g; }
#else
        if (den < 0) { num = neg(num); den = neg(den); }
        int64_t g = std::gcd(num < 0 ? neg(num) : num, den);
        if (g > 1) { num /= g; den /= g; }
#endif
    }

    static int64_t neg(int64_t x) {
        if (x == INT64_MIN) throw std::overflow_error("ed::Rational: int64 negation overflow");
        return -x;
    }
    static int64_t wide_to_i64(__int128 v, const char* what) {
        if (v < INT64_MIN || v > INT64_MAX)
            throw std::overflow_error(std::string("ed::Rational: ") + what);
        return (int64_t)v;
    }

    bool is_zero() const { return num == 0; }
    bool is_one() const { return num == 1 && den == 1; }

    Rational operator-() const {
        Rational r(*this);
        r.num = -r.num;
        return r;
    }
    Rational operator+() const { return *this; }

    Rational operator+(const Rational& o) const {
#ifdef ED_HAS_GMP
        return Rational(num * o.den + o.num * den, den * o.den);
#else
        // Cross-multiply in 128-bit; keeps int64 fast path without overflow.
        __int128 n = (__int128)num * o.den + (__int128)o.num * den;
        __int128 d = (__int128)den * o.den;
        return from_wide(n, d);
#endif
    }
    Rational operator-(const Rational& o) const { return *this + (-o); }
    Rational operator*(const Rational& o) const {
#ifdef ED_HAS_GMP
        return Rational(num * o.num, den * o.den);
#else
        return from_wide((__int128)num * o.num, (__int128)den * o.den);
#endif
    }
    Rational operator/(const Rational& o) const {
        if (o.num == 0) throw std::invalid_argument("ed::Rational: division by zero");
#ifdef ED_HAS_GMP
        return Rational(num * o.den, den * o.num);
#else
        return from_wide((__int128)num * o.den, (__int128)den * o.num);
#endif
    }

    Rational& operator+=(const Rational& o) { return *this = *this + o; }
    Rational& operator-=(const Rational& o) { return *this = *this - o; }
    Rational& operator*=(const Rational& o) { return *this = *this * o; }

    bool operator==(const Rational& o) const { return num == o.num && den == o.den; }
    bool operator!=(const Rational& o) const { return !(*this == o); }

    double to_double() const {
#ifdef ED_HAS_GMP
        return num.get_d() / den.get_d();
#else
        return (double)num / (double)den;
#endif
    }

    // Bit length of |num| / den (0 -> 0). Growth instrumentation shared by
    // both backends (NTReport).
    long long num_bits() const { return bits_of(num); }
    long long den_bits() const { return bits_of(den); }
    static long long bits_of(const CInt& v) {
#ifdef ED_HAS_GMP
        return (long long)mpz_sizeinbase(v.get_mpz_t(), 2);
#else
        uint64_t x = v < 0 ? (uint64_t)(-(v + 1)) + 1u : (uint64_t)v;
        return x ? 64 - __builtin_clzll(x) : 0;
#endif
    }
    // Small-value accessors (Phi coefficients, table comparisons; values in
    // these uses are always far below 2^31).
    int64_t num_i64() const {
#ifdef ED_HAS_GMP
        return (int64_t)num.get_si();
#else
        return num;
#endif
    }
    int64_t den_i64() const {
#ifdef ED_HAS_GMP
        return (int64_t)den.get_si();
#else
        return den;
#endif
    }
    std::string str() const {
#ifdef ED_HAS_GMP
        return num.get_str() + "/" + den.get_str();
#else
        return std::to_string(num) + "/" + std::to_string(den);
#endif
    }

  private:
#ifndef ED_HAS_GMP
    // int64 backend only: reduce before narrowing so intermediates like
    // (2^40 * 2^40) / (2^40 * 2^40) still come back representable.
    static Rational from_wide(__int128 n, __int128 d) {
        if (d == 0) throw std::invalid_argument("ed::Rational: zero denominator");
        if (d < 0) { n = -n; d = -d; }
        __int128 a = n < 0 ? -n : n, b = d;
        while (b != 0) { __int128 t = a % b; a = b; b = t; }
        if (a > 1) { n /= a; d /= a; }
        Rational r;
        r.num = (int64_t)n;
        r.den = (int64_t)d;
        if ((__int128)r.num != n || (__int128)r.den != d)
            throw std::overflow_error("ed::Rational: int64 overflow after reduction");
        return r;
    }
#endif
};

}  // namespace ed
