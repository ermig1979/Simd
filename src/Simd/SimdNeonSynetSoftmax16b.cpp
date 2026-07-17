/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE float32x4_t LoadBFloat16(const uint16_t* src)
        {
            return BFloat16ToFloat32(vmovl_u16(vld1_u16(src)));
        }

        SIMD_INLINE void StoreBFloat16(float32x4_t value, uint16_t* dst)
        {
            vst1_u16(dst, vmovn_u32(Float32ToBFloat16(value)));
        }

        void SynetSoftmax16b21(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            size_t aligned = Simd::AlignLo(outer, F), o = 0;
            for (; o < aligned; o += F)
            {
                uint16x4x2_t s = vld2_u16(src);
                float32x4_t src0 = BFloat16ToFloat32(vmovl_u16(s.val[0]));
                float32x4_t src1 = BFloat16ToFloat32(vmovl_u16(s.val[1]));
                float32x4_t max = vmaxq_f32(src0, src1);
                float32x4_t exp0 = exp.Exponent(vsubq_f32(src0, max));
                float32x4_t exp1 = exp.Exponent(vsubq_f32(src1, max));
                float32x4_t sum = vaddq_f32(exp0, exp1);
                uint16x4x2_t d;
                d.val[0] = vmovn_u32(Float32ToBFloat16(Div<1>(exp0, sum)));
                d.val[1] = vmovn_u32(Float32ToBFloat16(Div<1>(exp1, sum)));
                vst2_u16(dst, d);
                src += DF;
                dst += DF;
            }
            for (; o < outer; ++o)
            {
                float src0 = Base::BFloat16ToFloat32(src[0]);
                float src1 = Base::BFloat16ToFloat32(src[1]);
                float max = Simd::Max(src0, src1);
                float exp0 = ::exp(src0 - max);
                float exp1 = ::exp(src1 - max);
                float sum = exp0 + exp1;
                dst[0] = Base::Float32ToBFloat16(exp0 / sum);
                dst[1] = Base::Float32ToBFloat16(exp1 / sum);
                src += 2;
                dst += 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftmax16b31(const Exp& exp, float32x4_t buf[3])
        {
            float32x4_t max = vmaxq_f32(buf[0], vmaxq_f32(buf[1], buf[2]));
            buf[0] = exp.Exponent(vsubq_f32(buf[0], max));
            buf[1] = exp.Exponent(vsubq_f32(buf[1], max));
            buf[2] = exp.Exponent(vsubq_f32(buf[2], max));
            float32x4_t sum = vaddq_f32(buf[0], vaddq_f32(buf[1], buf[2]));
            buf[0] = Div<1>(buf[0], sum);
            buf[1] = Div<1>(buf[1], sum);
            buf[2] = Div<1>(buf[2], sum);
        }

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, float32x4_t dst[3])
        {
            uint16x4x3_t s = vld3_u16(src);
            dst[0] = BFloat16ToFloat32(vmovl_u16(s.val[0]));
            dst[1] = BFloat16ToFloat32(vmovl_u16(s.val[1]));
            dst[2] = BFloat16ToFloat32(vmovl_u16(s.val[2]));
        }

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, size_t size, float32x4_t dst[3])
        {
            SIMD_ALIGNED(16) uint16_t buf[A];
            for (size_t i = 0; i < size; i += 1)
            {
                buf[0 * F + i] = src[i * 3 + 0];
                buf[1 * F + i] = src[i * 3 + 1];
                buf[2 * F + i] = src[i * 3 + 2];
            }
            dst[0] = LoadBFloat16(buf + 0 * F);
            dst[1] = LoadBFloat16(buf + 1 * F);
            dst[2] = LoadBFloat16(buf + 2 * F);
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const float32x4_t src[3], uint16_t* dst)
        {
            uint16x4x3_t d;
            d.val[0] = vmovn_u32(Float32ToBFloat16(src[0]));
            d.val[1] = vmovn_u32(Float32ToBFloat16(src[1]));
            d.val[2] = vmovn_u32(Float32ToBFloat16(src[2]));
            vst3_u16(dst, d);
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const float32x4_t src[3], size_t size, uint16_t* dst)
        {
            SIMD_ALIGNED(16) uint16_t buf[A];
            StoreBFloat16(src[0], buf + 0 * F);
            StoreBFloat16(src[1], buf + 1 * F);
            StoreBFloat16(src[2], buf + 2 * F);
            for (size_t i = 0; i < size; i += 1)
            {
                dst[i * 3 + 0] = buf[0 * F + i];
                dst[i * 3 + 1] = buf[1 * F + i];
                dst[i * 3 + 2] = buf[2 * F + i];
            }
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            float32x4_t buf[3];
            size_t aligned = Simd::AlignLo(outer, F), o = 0;
            for (; o < aligned; o += F)
            {
                SynetSoftmax16b31Load(src, buf);
                SynetSoftmax16b31(exp, buf);
                SynetSoftmax16b31Save(buf, dst);
                src += 3 * F;
                dst += 3 * F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                SynetSoftmax16b31Load(src, tail, buf);
                SynetSoftmax16b31(exp, buf);
                SynetSoftmax16b31Save(buf, tail, dst);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void LoadTansp4x4(const uint16_t* src, size_t count, float* dst, float32x4_t& max)
        {
            float32x4_t a0 = LoadBFloat16(src + 0 * count);
            float32x4_t a1 = LoadBFloat16(src + 1 * count);
            float32x4_t a2 = LoadBFloat16(src + 2 * count);
            float32x4_t a3 = LoadBFloat16(src + 3 * count);

            float32x4x2_t b0 = vtrnq_f32(a0, a1);
            float32x4x2_t b1 = vtrnq_f32(a2, a3);
            a0 = vcombine_f32(vget_low_f32(b0.val[0]), vget_low_f32(b1.val[0]));
            a1 = vcombine_f32(vget_low_f32(b0.val[1]), vget_low_f32(b1.val[1]));
            a2 = vcombine_f32(vget_high_f32(b0.val[0]), vget_high_f32(b1.val[0]));
            a3 = vcombine_f32(vget_high_f32(b0.val[1]), vget_high_f32(b1.val[1]));

            max = vmaxq_f32(max, a0);
            max = vmaxq_f32(max, a1);
            max = vmaxq_f32(max, a2);
            max = vmaxq_f32(max, a3);

            Store<false>(dst + 0 * F, a0);
            Store<false>(dst + 1 * F, a1);
            Store<false>(dst + 2 * F, a2);
            Store<false>(dst + 3 * F, a3);
        }

        SIMD_INLINE void StoreTansp4x4(const float* src, float32x4_t k, uint16_t* dst, size_t count)
        {
            float32x4_t a0 = vmulq_f32(Load<false>(src + 0 * F), k);
            float32x4_t a1 = vmulq_f32(Load<false>(src + 1 * F), k);
            float32x4_t a2 = vmulq_f32(Load<false>(src + 2 * F), k);
            float32x4_t a3 = vmulq_f32(Load<false>(src + 3 * F), k);

            float32x4x2_t b0 = vtrnq_f32(a0, a1);
            float32x4x2_t b1 = vtrnq_f32(a2, a3);
            a0 = vcombine_f32(vget_low_f32(b0.val[0]), vget_low_f32(b1.val[0]));
            a1 = vcombine_f32(vget_low_f32(b0.val[1]), vget_low_f32(b1.val[1]));
            a2 = vcombine_f32(vget_high_f32(b0.val[0]), vget_high_f32(b1.val[0]));
            a3 = vcombine_f32(vget_high_f32(b0.val[1]), vget_high_f32(b1.val[1]));

            StoreBFloat16(a0, dst + 0 * count);
            StoreBFloat16(a1, dst + 1 * count);
            StoreBFloat16(a2, dst + 2 * count);
            StoreBFloat16(a3, dst + 3 * count);
        }

        void SynetSoftmax16bX1(const uint16_t* src, size_t outer, size_t count, uint16_t* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                for (c = 0; c < countF; c += F)
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                if (c < count)
                {
                    c = count - F;
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                }
                float32x4_t _sum = vdupq_n_f32(0.0f);
                for (size_t c = 0; c < count; ++c)
                {
                    float32x4_t _exp = exp.Exponent(vsubq_f32(Load<false>(buf.data + c * F), _max));
                    _sum = vaddq_f32(_sum, _exp);
                    Store<false>(buf.data + c * F, _exp);
                }
                float32x4_t _k = Div<1>(vdupq_n_f32(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp4x4(buf.data + c * F, _k, dst + c, count);
                if (c < count)
                {
                    c = count - F;
                    StoreTansp4x4(buf.data + c * F, _k, dst + c, count);
                }
                src += count * F;
                dst += count * F;
            }
            for (; o < outer; ++o)
            {
                for (size_t c = 0; c < count; ++c)
                    buf[c] = Base::BFloat16ToFloat32(src[c]);

                float max = buf[0];
                for (size_t c = 1; c < count; ++c)
                    max = Simd::Max(max, buf[c]);
                float sum = 0;
                for (size_t c = 0; c < count; ++c)
                {
                    buf[c] = ::exp(buf[c] - max);
                    sum += buf[c];
                }
                float k = 1.0f / sum;
                for (size_t c = 0; c < count; ++c)
                    dst[c] = Base::Float32ToBFloat16(buf[c] * k);
                src += count;
                dst += count;
            }
        }

        void SynetSoftmax16b(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                    SynetSoftmax16b21(src, outer, dst);
                else if (count == 3)
                    SynetSoftmax16b31(src, outer, dst);
                else
                    SynetSoftmax16bX1(src, outer, count, dst);
            }
            else
            {
                Exp exp;
                size_t innerF = Simd::AlignLo(inner, F);
                Array32f _buf(inner * (count + 2));
                float* max = _buf.data, * sum = _buf.data + inner, * buf = sum + inner, * b;
                for (size_t o = 0; o < outer; ++o)
                {
                    BFloat16ToFloat32(src, count * inner, buf);
                    memcpy(max, buf, inner * sizeof(float));
                    b = buf + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                            Store<false>(max + i, vmaxq_f32(Load<false>(b + i), Load<false>(max + i)));
                        for (; i < inner; ++i)
                            max[i] = Simd::Max(max[i], b[i]);
                        b += inner;
                    }

                    b = buf;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                        {
                            float32x4_t _d = exp.Exponent(vsubq_f32(Load<false>(b + i), Load<false>(max + i)));
                            Store<false>(b + i, _d);
                            Store<false>(sum + i, vaddq_f32(_d, Load<false>(sum + i)));
                        }
                        for (; i < inner; ++i)
                        {
                            b[i] = ::exp(b[i] - max[i]);
                            sum[i] += b[i];
                        }
                        b += inner;
                    }

                    b = buf;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                            StoreBFloat16(Div<1>(Load<false>(b + i), Load<false>(sum + i)), dst + i);
                        for (; i < inner; ++i)
                            dst[i] = Base::Float32ToBFloat16(b[i] / sum[i]);
                        b += inner;
                        dst += inner;
                    }
                    src += count * inner;
                }
            }
        }
    }
#endif
}
