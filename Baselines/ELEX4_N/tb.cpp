// 通用 verilator 测试台：读取 "a b expected" 三元组文件, 驱动 DUT, 逐一对比。
// 编译期宏: DUT_HEADER(头), DUT_CLASS(类), VEC_FILE(向量文件)。
#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(DUT_HEADER)
#include <verilated.h>
#include <cstdio>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    DUT_CLASS* dut = new DUT_CLASS;
    FILE* f = fopen(STR(VEC_FILE), "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", STR(VEC_FILE)); return 2; }
    unsigned long a, b, exp;
    long n = 0, mism = 0;
    while (fscanf(f, "%lu %lu %lu", &a, &b, &exp) == 3) {
        dut->a = a; dut->b = b; dut->eval();
        unsigned long got = (unsigned long)dut->p;
        ++n;
        if (got != exp) {
            if (mism < 3)
                fprintf(stderr, "MISMATCH a=%lu b=%lu got=%lu exp=%lu\n", a, b, got, exp);
            ++mism;
        }
    }
    fclose(f); delete dut;
    if (mism == 0) { printf("PASS: all %ld vectors match golden\n", n); return 0; }
    printf("FAIL: %ld/%ld mismatches\n", mism, n); return 1;
}
