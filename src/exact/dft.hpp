// exact-dft — L5: all-lengths driver + Bluestein chirp (paper sec 5).
//
// Pipeline for arbitrary n (mirrors the paper's architecture):
//   1. Working length: L = R_ell * 2^e in [2n, 4n) (lengths.hpp), radices
//      {3, 5, 7, ..., 2^e} pairwise coprime.
//   2. Length-L transform over the composite ring Q(zeta_D), D = lcm(2n, L):
//      the odd-prime axes are compiled by the L2 exact-width compiler and
//      scheduled by the L3 traversal (TensorFourier with exec ring D); the
//      power-of-two axis is a standard exact radix-2 DIT butterfly in K
//      (the paper's power-of-two fill). The axes are combined by the
//      coprime-index (Good–Thomas) identity, derived exactly as in the
//      paper's prime-product structure:
//          zeta_L^{jk} = prod_i rho_i^{t_i * j_i * k_i},
//          rho_i = zeta_D^{D/r_i},  t_i = (L/r_i)^{-1} mod r_i,
//          j_i = j mod r_i,
//      so F_L acts as the tensor product of the F_{r_i}(rho_i) after the
//      input digit map a_i = t_i * j_i (mod r_i); outputs read at digit
//      b_i = k_i directly. No twiddle factors between axes (that coprimality
//      is exactly why the working length is a prime product).
//   3. Bluestein chirp (spec judgment call #4, documented deviation):
//      with w a primitive 2n-th root (w^2 = zeta_n element), jk =
//      (j^2 + k^2 - (k-j)^2)/2 gives
//          y_k = w^{k^2} * sum_j (x_j w^{j^2}) * w^{-(k-j)^2},
//      a chirp multiplication, a convolution at length L (circular = linear
//      here because L >= 2n > 2n - 2), and a de-chirp. The convolution uses
//      two forward length-L transforms, pointwise multiply, index negation,
//      one more length-L transform and a 1/L scale:
//          IDFT_L(z)_k = (1/L) * DFT_L(negidx z)_k.
//      The paper formulates the chirp at the working length; we chirp with
//      zeta_{2n} directly — algebraically equivalent, exactness unaffected.
//      The chirp kernel's length-L transform is compiled once per length and
//      cached across dft() calls.
//
// exact_convolve: Corollary 1.3 flavor — linear convolution via three
// length-L transforms (no chirp needed), ring lcm(input rings, L).
//
// direct_dft: the same coprime-axis machinery applied at length n itself
// (axes = prime powers of n, ring n) — the "direct composite-ring path"
// used as the differential counterpart of the chirp path (spec L5 tests).
//
// Runtime honesty: ring multiplication is dense O(phi(D)^2) rational ops,
// and phi(lcm(2n, L)) grows quickly with the part of n coprime to L — the
// practical exact sizes are limited by this, not by register width (GMP
// removes the width limit only). See test_driver size notes.
#pragma once
#include "cyclotomic.hpp"
#include "lengths.hpp"
#include "newton_toeplitz.hpp"
#include "oracle_dft.hpp"
#include "schedule.hpp"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ed {

// Modular inverse (gcd(a, m) = 1, m > 0), extended Euclid.
inline int64_t inv_mod(int64_t a, int64_t m) {
    if (m <= 0) throw std::invalid_argument("inv_mod: m must be > 0");
    int64_t aa = a % m;
    if (aa < 0) aa += m;
    int64_t prev = 0, cur = 1, b = m;
    while (aa != 0) {
        const int64_t q = b / aa;
        const int64_t tp = prev - q * cur; prev = cur; cur = tp;
        const int64_t tb = b - q * aa; b = aa; aa = tb;
    }
    if (b != 1) throw std::logic_error("inv_mod: arguments not coprime");
    return prev < 0 ? prev + m : prev;
}

// Exact radix-2 DIT: F_N(rho) * a in place, N = a.size() a power of two,
// rho a primitive N-th root. y_k = sum_j a_j rho^{jk}.
inline void radix2_dft(std::vector<Cyclotomic>& a, const Cyclotomic& rho) {
    const size_t n = a.size();
    if (n <= 1) return;
    std::vector<Cyclotomic> ev(n / 2), od(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        ev[i] = std::move(a[2 * i]);
        od[i] = std::move(a[2 * i + 1]);
    }
    const Cyclotomic rho2 = rho * rho;
    radix2_dft(ev, rho2);
    radix2_dft(od, rho2);
    const int d = rho.d();
    Cyclotomic w = Cyclotomic::integer(d, 1);
    for (size_t j = 0; j < n / 2; ++j) {
        const Cyclotomic t = w * od[j];
        a[j] = ev[j] + t;
        a[j + n / 2] = ev[j] - t;
        w = w * rho;
    }
}

// ---------------------------------------------------------------------------
// CoprimeTransform: F_L over ring D with axes [2^e, q_1, ..., q_m]
// (pairwise coprime, q_i odd prime powers), prod = L, L | D.
// Odd axes via L2+L3 (TensorFourier, exec ring D); 2^e axis via radix-2.
// ---------------------------------------------------------------------------
class CoprimeTransform {
  public:
    CoprimeTransform() = default;  // uninitialized L=1 identity (ExactDFT member init)
    CoprimeTransform(std::vector<int> odd_axes, int two_exp, int ring)
        : e_(two_exp), D_(ring) {
        if (e_ < 0) throw std::invalid_argument("CoprimeTransform: bad 2-exp");
        q_.clear();
        for (int r : odd_axes)
            if (r > 1) q_.push_back(r);
        A2_ = 1 << e_;
        R_ = 1;
        for (int r : q_) {
            if (r < 2 || (r & 1) == 0)
                throw std::invalid_argument("CoprimeTransform: odd axes only");
            R_ *= r;
        }
        L_ = A2_ * R_;
        if (D_ % L_ != 0 || D_ < 1)
            throw std::invalid_argument("CoprimeTransform: ring must be a multiple of L");
        // Per-axis root alignment: t_i = (L/r_i)^{-1} mod r_i; the power-of-two
        // axis root is rho_A = zeta_D^{D/A2}.
        tau_.assign(q_.size() + 1, 1);
        for (int i = (int)q_.size() - 1; i >= 0; --i) tau_[i] = tau_[i + 1] * q_[i];
        tq_.resize(q_.size());
        for (size_t i = 0; i < q_.size(); ++i)
            tq_[i] = (int)inv_mod((L_ / q_[i]) % q_[i], q_[i]);
        tA_ = e_ > 0 ? (int)inv_mod(R_ % A2_, A2_) : 0;  // R_ odd => coprime to 2^e
        if (e_ > 0) rhoA_ = Cyclotomic::zeta(D_).pow(D_ / A2_);
        if (!q_.empty()) tf_.emplace(q_, D_);
        build_perms();
    }

    int L() const { return L_; }
    int ring() const { return D_; }
    long long applies() const { return applies_; }
    long long radix2_mults() const { return radix2_mults_; }
    const TensorFourier::Report& axes_report() const { return tf_report_; }

    // x (length L, ring D) -> F_L(x) in place.
    void apply(std::vector<Cyclotomic>& x) {
        if ((int)x.size() != L_)
            throw std::invalid_argument("CoprimeTransform::apply: length != L");
        for (const Cyclotomic& c : x)
            if (c.d() != D_)
                throw std::invalid_argument("CoprimeTransform::apply: ring mismatch");
        ++applies_;
        std::vector<Cyclotomic> buf(L_, Cyclotomic(D_));
        for (int j = 0; j < L_; ++j) buf[in_[j]] = std::move(x[j]);
        // Power-of-two axis (stride R_).
        if (e_ > 0) {
            std::vector<Cyclotomic> col(A2_);
            for (int b = 0; b < R_; ++b) {
                for (int a = 0; a < A2_; ++a) col[a] = std::move(buf[a * R_ + b]);
                radix2_dft(col, rhoA_);
                radix2_mults_ += A2_ * (e_ + 1) / 2;  // butterfly+twiddle mults
                for (int a = 0; a < A2_; ++a) buf[a * R_ + b] = std::move(col[a]);
            }
        }
        // Odd prime-power axes: contiguous R_-blocks, L3 scheduled words.
        if (tf_) {
            std::vector<Cyclotomic> sub(R_);
            for (int a = 0; a < A2_; ++a) {
                for (int r = 0; r < R_; ++r) sub[r] = std::move(buf[a * R_ + r]);
                tf_->apply(sub);
                for (int r = 0; r < R_; ++r) buf[a * R_ + r] = std::move(sub[r]);
            }
            tf_report_ = tf_->report();
        }
        for (int k = 0; k < L_; ++k) x[k] = std::move(buf[out_[k]]);
    }

  private:
    void build_perms() {
        in_.assign(L_, 0);
        out_.assign(L_, 0);
        for (int j = 0; j < L_; ++j) {
            // CRT residues j_i = j mod r_i (NOT mixed-radix digits), then
            // root-aligned input digits a_i = t_i * j_i mod r_i.
            int fin = e_ > 0 ? (int)((int64_t)tA_ * (j % A2_) % A2_) * R_ : 0;
            int fout = e_ > 0 ? (j % A2_) * R_ : 0;
            for (size_t i = 0; i < q_.size(); ++i) {
                const int di = j % q_[i];
                fin += (int)((int64_t)tq_[i] * di % q_[i]) * tau_[i + 1];
                fout += di * tau_[i + 1];
            }
            in_[j] = fin;
            out_[j] = fout;
        }
    }

    std::vector<int> q_;
    int e_ = 0, A2_ = 1, R_ = 1, L_ = 1, D_ = 1, tA_ = 0;
    std::vector<int> tau_, tq_, in_, out_;
    std::optional<TensorFourier> tf_;
    Cyclotomic rhoA_{1};
    long long applies_ = 0, radix2_mults_ = 0;
    TensorFourier::Report tf_report_;
};

// ---------------------------------------------------------------------------
// ExactDFT: the all-lengths driver (chirp path).
// ---------------------------------------------------------------------------
struct DriverReport {
    int n = 0, L = 0, ring = 0, two_exp = 0;
    std::vector<int> odd_primes;
    long long transform_calls = 0;  // length-L scheduled transform applications
    long long radix2_mults = 0;     // power-of-two axis ring multiplications
    TensorFourier::Report axes;     // odd-axis compiled-word reports
};

class ExactDFT {
  public:
    // ring_multiple: further factor folded into the transform ring D (must
    // be a positive integer). The Python surface uses this to host i (ring
    // factor 4) for rational-complex inputs at odd working lengths.
    explicit ExactDFT(int n, int ring_multiple = 1) : n_(n), wl_(working_length(n)) {
        if (n_ < 1) throw std::invalid_argument("ExactDFT: n >= 1");
        if (ring_multiple < 1)
            throw std::invalid_argument("ExactDFT: ring_multiple >= 1");
        L_ = wl_.L;
        long long D = std::lcm(2LL * n_, (long long)L_) * 1LL;
        D = std::lcm(D, (long long)ring_multiple);
        if (D > kMaxCyclotomicOrder)
            throw std::invalid_argument("ExactDFT: ring order exceeds Phi cap");
        D_ = (int)D;
        ct_ = CoprimeTransform(wl_.odd_primes, wl_.two_exp, D_);
        w_ = Cyclotomic::zeta(D_).pow(D_ / (2LL * n_));  // w^2 = zeta_n element
        // Chirp kernel B (B[m mod L] = w^{-m^2}, m in (-(n-1), n-1)) and its
        // cached length-L transform.
        B_.assign(L_, Cyclotomic(D_));
        B_[0] = Cyclotomic::integer(D_, 1);
        Cyclotomic run = Cyclotomic::integer(D_, 1);
        for (int m = 1; m < n_; ++m) {
            // w^{-(m^2 - (m-1)^2)} = w^{-(2m-1)} = w^{2n - 2m + 1}
            run = run * w_.pow((2LL * n_ - (2LL * m - 1)) % (2LL * n_));
            B_[m] = run;
            B_[L_ - m] = run;
        }
        ct_.apply(B_);  // Bhat cached
    }

    int n() const { return n_; }
    int ring() const { return D_; }
    int working_l() const { return L_; }
    const DriverReport report() const {
        DriverReport r;
        r.n = n_; r.L = L_; r.ring = D_; r.two_exp = wl_.two_exp;
        r.odd_primes = wl_.odd_primes;
        r.transform_calls = ct_.applies();
        r.radix2_mults = ct_.radix2_mults();
        r.axes = ct_.axes_report();
        return r;
    }

    // Exact F_n * x. Inputs: length n, any ring d with d | D (ring n and D
    // both fine); outputs in ring D (compare with the oracle after
    // ed::embed — injective ring homomorphism, bit-exact).
    std::vector<Cyclotomic> dft(std::vector<Cyclotomic> x) {
        if ((int)x.size() != n_) throw std::invalid_argument("dft: length != n");
        for (Cyclotomic& c : x)
            if (c.d() != D_) c = embed(c, D_);
        // Chirp in: A_j = x_j * w^{j^2}, zero pad to L.
        std::vector<Cyclotomic> A(L_, Cyclotomic(D_));
        {
            Cyclotomic run = Cyclotomic::integer(D_, 1);
            for (int j = 0; j < n_; ++j) {
                A[j] = x[j] * run;
                run = run * w_.pow((2LL * j + 1) % (2LL * n_));  // w^{(j+1)^2 - j^2}
            }
        }
        ct_.apply(A);
        for (int k = 0; k < L_; ++k) A[k] = A[k] * B_[k];
        // IDFT_L via index negation: u_k = U_{(-k) mod L}, C = (1/L) F_L(u).
        std::reverse(A.begin() + 1, A.end());
        ct_.apply(A);
        // De-chirp: y_k = w^{k^2} * C_k with C = (1/L) * F_L(negidx U).
        const Cyclotomic invL = Cyclotomic::rational(D_, Rational(1) / Rational(L_));
        std::vector<Cyclotomic> y(n_, Cyclotomic(D_));
        {
            Cyclotomic run = Cyclotomic::integer(D_, 1);
            for (int k = 0; k < n_; ++k) {
                y[k] = run * A[k] * invL;
                run = run * w_.pow((2LL * k + 1) % (2LL * n_));
            }
        }
        return y;
    }

  private:
    int n_ = 0, L_ = 0, D_ = 1;
    WorkingLength wl_;
    CoprimeTransform ct_;
    Cyclotomic w_{1};
    std::vector<Cyclotomic> B_;  // cached chirp-kernel length-L transform
};

// Convenience: one-shot exact DFT (compilation per call).
inline std::vector<Cyclotomic> exact_dft(int n, const std::vector<Cyclotomic>& x) {
    ExactDFT drv(n);
    return drv.dft(x);
}

// ---------------------------------------------------------------------------
// Direct composite-ring path: F_n with axes = prime powers of n (pairwise
// coprime), ring n, radix-2 butterfly on the 2-part. Differential
// counterpart of the chirp path (spec L5).
// ---------------------------------------------------------------------------
inline std::vector<Cyclotomic> direct_dft(int n, const std::vector<Cyclotomic>& x) {
    if (n < 1 || (int)x.size() != n)
        throw std::invalid_argument("direct_dft: length mismatch");
    if (n == 1) return x;
    std::vector<int> odd_pp;
    int rest = n;
    int e = 0;
    while (rest % 2 == 0) { rest /= 2; ++e; }
    for (int p = 3; p * p <= rest || rest > 1; p += 2) {
        if (p * p > rest) break;
        if (rest % p == 0) {
            int q = 1;
            while (rest % p == 0) { rest /= p; q *= p; }
            odd_pp.push_back(q);
        }
    }
    if (rest > 1) odd_pp.push_back(rest);
    CoprimeTransform ct(odd_pp, e, n);
    auto y = x;
    for (const Cyclotomic& c : y)
        if (c.d() != n) throw std::invalid_argument("direct_dft: inputs must be in ring n");
    ct.apply(y);
    return y;
}

// Exact linear convolution via three length-L transforms (working length for
// n = la + lb - 1). Ring = lcm(input rings, L); no chirp roots needed.
inline std::vector<Cyclotomic> exact_convolve(const std::vector<Cyclotomic>& a,
                                              const std::vector<Cyclotomic>& b) {
    if (a.empty() || b.empty())
        throw std::invalid_argument("exact_convolve: empty input");
    if (a[0].d() != b[0].d())
        throw std::invalid_argument("exact_convolve: rings must match");
    const int n = (int)(a.size() + b.size() - 1);
    const WorkingLength wl = working_length(n);
    const int L = wl.L;
    const int D = (int)std::lcm((long long)a[0].d(), (long long)L);
    if (D > kMaxCyclotomicOrder)
        throw std::invalid_argument("exact_convolve: ring order exceeds Phi cap");
    CoprimeTransform ct(wl.odd_primes, wl.two_exp, D);
    std::vector<Cyclotomic> A(L, Cyclotomic(D)), Bv(L, Cyclotomic(D));
    for (size_t i = 0; i < a.size(); ++i) A[i] = embed(a[i], D);
    for (size_t i = 0; i < b.size(); ++i) Bv[i] = embed(b[i], D);
    ct.apply(A);
    ct.apply(Bv);
    for (int k = 0; k < L; ++k) A[k] = A[k] * Bv[k];
    std::reverse(A.begin() + 1, A.end());
    ct.apply(A);
    A.resize(n);
    const Cyclotomic invL = Cyclotomic::rational(D, Rational(1) / Rational(L));
    for (auto& c : A) c = c * invL;
    return A;
}

}  // namespace ed
