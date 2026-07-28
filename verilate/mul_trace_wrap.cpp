// Deterministic per-vector trace harness for semantic cross-checking.
//
// Uses the same xorshift128+ stream as mul_err_wrap.cpp and prints:
//   index,a,b,out31
//
// This harness is intentionally not used by training.  It exists only to
// prove that the hard tensor simulator and emitted RTL produce identical
// outputs for every checked input vector.
#include "VMUL.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

static const uint64_t SEED = 12345ULL;
static const uint32_t MASK31 = 0x7FFFFFFFu;

int main(int argc, char** argv) {
    long long count = 4096;
    if (argc > 1) {
        const long long value = atoll(argv[1]);
        if (value > 0) count = value;
    }
    Verilated::commandArgs(argc, argv);
    auto top = std::make_shared<VMUL>();

    uint64_t s0 = 0x9E3779B97F4A7C15ULL ^ SEED;
    uint64_t s1 = 0xD1B54A32D192ED03ULL;
    for (long long i = 0; i < count; ++i) {
        uint64_t x = s0;
        const uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        const uint64_t random = s1 + y;
        const uint16_t a = static_cast<uint16_t>(random & 0xFFFF);
        const uint16_t b = static_cast<uint16_t>((random >> 16) & 0xFFFF);

        top->clk = 0;
        top->a = a;
        top->b = b;
        top->eval();
        const uint32_t out = static_cast<uint32_t>(top->out) & MASK31;
        std::printf(
            "%lld,%u,%u,%u\n",
            i,
            static_cast<unsigned>(a),
            static_cast<unsigned>(b),
            static_cast<unsigned>(out)
        );
    }
    return 0;
}
