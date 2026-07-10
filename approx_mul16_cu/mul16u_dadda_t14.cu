// ============================================================================
// mul16u_dadda_t14.cu
// 16x16 近似乘法器 -- 纯截断 Dadda 树 (truncated Dadda multiplier)
//
// 结构: 16x16 部分积矩阵 (32 列), 最低 14 列 (列 0..13) 的部分积直接
//       丢弃(不生成、不补偿), 其余部分积用标准 Dadda 树精确压缩。
//       由于保留部分的归约是精确的, 功能上逐位等价于:
//         P = sum_{i=0..15} a_i ? ((b << i) & (0xFFFFFFFF << 14)) : 0
//
// 无符号核心 MRED = 6.0e-04 (2^24 组均匀随机 uint16 输入), 误差恒为非负。
//
// 接口: int32_t int_mul(int32_t, int32_t), 与工具约定一致。
// 入参按有符号数处理(符号-幅值), 非负输入(如 uint 量化码)原样进核心;
// 要求 |a|,|b| <= 65535, 更高位被 16bit 核心自然截掉。
// 注意: 若两个无符号码都接近 65535, 精确积超出 int32 正范围,
// 返回值按 32 位补码回绕(与工具 8bit 用法不冲突, 16bit uint 满幅时留意)。
// ============================================================================

namespace app {
#include <cstdint>

// 无符号核心: 纯截断 Dadda, 丢弃部分积第 0..13 列
__device__ uint32_t dadda16u_trunc(uint16_t a, uint16_t b)
{
    const uint32_t keep = 0xFFFFFFFFu << 14;
    uint32_t p = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        if ((a >> i) & 1u)
            p += ((uint32_t)b << i) & keep;
    return p;
}

__device__ int32_t int_mul(int32_t a, int32_t b)
{
    uint8_t a_sign, b_sign;
    uint16_t a_mul, b_mul;
    int32_t mul;

    if (a < 0)
    {
        a_sign = 1;
        a_mul = (uint16_t)(-(int64_t)a);
    }
    else
    {
        a_sign = 0;
        a_mul = (uint16_t)a;
    }
    if (b < 0)
    {
        b_sign = 1;
        b_mul = (uint16_t)(-(int64_t)b);
    }
    else
    {
        b_sign = 0;
        b_mul = (uint16_t)b;
    }

    mul = (int32_t)dadda16u_trunc(a_mul, b_mul);

    if (a_sign == b_sign)
    {
        return mul;
    }
    else
    {
        return (-1) * mul;
    }
}
}  // namespace app
