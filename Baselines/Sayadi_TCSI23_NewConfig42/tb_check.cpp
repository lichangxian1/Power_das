// RTL==golden checker for MUL(clk,a,b,out[30:0]). Reads "a b golden31" lines.
#include "VMUL.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) { fprintf(stderr, "usage: %s vecfile\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("vec"); return 2; }
    VMUL top;
    unsigned long long a, b, g;
    long long n = 0, bad = 0;
    while (fscanf(f, "%llu %llu %llu", &a, &b, &g) == 3) {
        top.a = (uint32_t)a; top.b = (uint32_t)b; top.eval();
        uint32_t got = top.out & 0x7FFFFFFFu;
        if (got != (uint32_t)g) {
            if (bad < 10)
                printf("MISMATCH a=%llu b=%llu golden=%llu rtl=%u\n", a, b, g, got);
            bad++;
        }
        n++;
    }
    fclose(f);
    printf("%s: %lld vectors, %lld mismatches -> %s\n", argv[1], n, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
