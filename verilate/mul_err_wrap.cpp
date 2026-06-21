// In-loop Verilator MC error harness for the 16x16 AND-encoded MUL (31-bit [30:0] output).
// Used as the training error GATE (真实 MED) — replaces the analytic upper-bound proxy.
//
//   masked error e = circular-wrap( out[30:0] - (a*b & 0x7FFFFFFF) ) onto [-2^30, 2^30)
//   so a boundary crossing recovers the true minimal-magnitude approximation error.
//
// Fixed seed + fixed xorshift128+ stream => COMMON RANDOM NUMBERS across all designs
// (every design sees the identical vector sequence; smaller N is a prefix of larger N),
// so cross-design objective differences are NOT polluted by MC noise.
//
// Vector count = argv[1] if given, else 16M.
// Output (single line, parsed by trainer): "masked,MED,BIAS_signed,RMSE,ER_pct,MaxAbsErr"
//   - BIAS is the SIGNED mean error (objective takes abs() itself).
//   - MaxAbsErr (MC WCE) is reported but NOT trusted as a gate (MC tail does not converge).
#include "VMUL.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>

static const uint64_t SEED = 12345ULL;
static const uint32_t MASK31 = 0x7FFFFFFFu;

// circular wrap of a 31-bit-domain error onto [-2^30, 2^30)
static inline long long wrap31(long long e) {
    const long long HALF = 1LL << 30, FULL = 1LL << 31;
    if (e > HALF) e -= FULL; else if (e < -HALF) e += FULL;
    return e;
}

int main(int argc, char** argv) {
    long long MC = 16000000LL;
    if (argc > 1) { long long v = atoll(argv[1]); if (v > 0) MC = v; }
    Verilated::commandArgs(argc, argv);
    auto top = std::make_shared<VMUL>();

    __int128 abs_sum = 0;       // Σ|e|   (exact integer accumulation)
    __int128 sig_sum = 0;       // Σe     (signed)
    double   sq_sum  = 0.0;     // Σe^2   (double; RMSE is informational)
    long long n_err = 0, m_max = 0;

    uint64_t s0 = 0x9E3779B97F4A7C15ULL ^ SEED, s1 = 0xD1B54A32D192ED03ULL;
    for (long long i = 0; i < MC; i++) {
        uint64_t x = s0, y = s1; s0 = y;
        x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r = s1 + y;
        uint16_t a = (uint16_t)(r & 0xFFFF);
        uint16_t b = (uint16_t)((r >> 16) & 0xFFFF);

        top->clk = 0; top->a = a; top->b = b; top->eval();
        uint32_t out = (uint32_t)top->out & MASK31;
        uint32_t golden = ((uint32_t)a * (uint32_t)b) & MASK31;

        long long e = wrap31((long long)out - (long long)golden);
        long long ae = e < 0 ? -e : e;
        abs_sum += (__int128)ae;
        sig_sum += (__int128)e;
        sq_sum  += (double)e * (double)e;
        if (ae > m_max) m_max = ae;
        if (e != 0) n_err++;
    }

    double dn = (double)MC;
    double med  = (double)(long long)(abs_sum / MC) + (double)(long long)(abs_sum % MC) / dn;
    double bias = (double)(long long)(sig_sum / (__int128)MC)
                + (double)(long long)(sig_sum % (__int128)MC) / dn;
    double rmse = sqrt(sq_sum / dn);
    double er   = (double)n_err / dn * 100.0;

    printf("masked,%.6f,%.6f,%.6f,%.6f,%lld\n", med, bias, rmse, er, m_max);
    return 0;
}
