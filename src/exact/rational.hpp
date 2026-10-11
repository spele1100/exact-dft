// exact-dft — L0: exact rational arithmetic.
// int64 numerator/denominator, always gcd-reduced, denominator > 0.
// All intermediate arithmetic goes through __int128 with explicit overflow
// detection (throws std::overflow_error). GMP fallback is a build option
// planned for later milestones (see spec judgment call #5).
#pragma once
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ed {

struct Rational {
    int64_t num = 0;
    int64_t den = 1;

    constexpr Rational() = default;
    constexpr Rational(int64_t n) : num(n), den(1) {}
    Rational(int64_t n, int64_t d) : num(n), den(d) { normalize(); }

    void normalize() {
        if (den == 0) throw std::invalid_argument("ed::Rational: zero denominator");
        if (den < 0) { num = neg(num); den = neg(den); }
        int64_t g = std::gcd(num < 0 ? neg(num) : num, den);
        if (g > 1) { num /= g; den /= g; }
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

    Rational operator-() const { Rational r(*this); r.num = neg(r.num); return r; }
    Rational operator+() const { return *this; }

    Rational operator+(const Rational& o) const {
        // Cross-multiply in 128-bit; keeps int64 fast path without overflow.
        __int128 n = (__int128)num * o.den + (__int128)o.num * den;
        __int128 d = (__int128)den * o.den;
        return from_wide(n, d);
    }
    Rational operator-(const Rational& o) const { return *this + (-o); }
    Rational operator*(const Rational& o) const {
        return from_wide((__int128)num * o.num, (__int128)den * o.den);
    }
    Rational operator/(const Rational& o) const {
        if (o.num == 0) throw std::invalid_argument("ed::Rational: division by zero");
        return from_wide((__int128)num * o.den, (__int128)den * o.num);
    }

    Rational& operator+=(const Rational& o) { return *this = *this + o; }
    Rational& operator-=(const Rational& o) { return *this = *this - o; }
    Rational& operator*=(const Rational& o) { return *this = *this * o; }

    bool operator==(const Rational& o) const { return num == o.num && den == o.den; }
    bool operator!=(const Rational& o) const { return !(*this == o); }

    double to_double() const { return (double)num / (double)den; }

  private:
    static Rational from_wide(__int128 n, __int128 d) {
        if (d == 0) throw std::invalid_argument("ed::Rational: zero denominator");
        if (d < 0) { n = -n; d = -d; }
        // 128-bit Euclid; reduce before narrowing so intermediates like
        // (2^40 * 2^40) / (2^40 * 2^40) still come back representable.
        __int128 a = n < 0 ? -n : n, b = d;
        while (b != 0) { __int128 t = a % b; a = b; b = t; }
        if (a > 1) { n /= a; d /= a; }
        Rational r;
        r.num = (int64_t)n;  // checked by caller-visible constructor below
        r.den = (int64_t)d;
        if ((__int128)r.num != n || (__int128)r.den != d)
            throw std::overflow_error("ed::Rational: int64 overflow after reduction");
        return r;
    }
};

}  // namespace ed
