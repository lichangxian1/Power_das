// ============================================================================
// w11/mul16u_dadda_t11.cu
// 16x16 近似乘法器 -- 纯截断 Dadda 树, 按"激活16bit x 权重11bit"口径标定
//
// 结构与上级目录相同: 丢弃 16x16 部分积矩阵最低 11 列 (列 0..10),
// 其余部分积用标准 Dadda 树精确压缩。
//
// MRED = 1.5e-03  (口径: a ~ U[0,65535], b ~ U[0,2047], 2^24 样本;
// 权重只有 11bit 时乘积整体偏小, 同样的 K 相对误差比均匀 16x16 口径大,
// 故本目录的 K 档位与上级目录不同, 专用于工具权重位宽=11 的实验)
//
// 接口: int32_t int_mul(int32_t, int32_t), 固定签名。
// ============================================================================

namespace app {
#include <cstdint>

// 无符号核心: 纯截断 Dadda, 丢弃部分积第 0..10 列
__device__ uint32_t dadda16u_trunc(uint16_t a, uint16_t b)
{
    const uint32_t keep = 0xFFFFFFFFu << 11;
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
