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
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svfloat32_t BFloat16ToFloat32(svuint32_t value, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, value, Base::Bf16::SHIFT));
        }

        SIMD_INLINE svfloat32_t BFloat16ToFloat32(svuint16_t value, const svbool_t& mask)
        {
            return BFloat16ToFloat32(svunpklo_u32(value), mask);
        }

        SIMD_INLINE svuint16_t Float32ToBFloat16Lo(svfloat32_t value, const svbool_t& mask)
        {
            return svqxtnb_u32(Float32ToBFloat16(value, mask));
        }

        SIMD_INLINE svfloat32_t SoftmaxPoly5(const svbool_t& mask, svfloat32_t x)
        {
            svfloat32_t p = svdup_n_f32(1.8775767e-3f);
            p = svmla_f32_x(mask, svdup_n_f32(8.9893397e-3f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(5.5826318e-2f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(2.4015361e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(6.9315308e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(9.9999994e-1f), x, p);
            return p;
        }

        SIMD_INLINE svfloat32_t SoftmaxExp2(const svbool_t& mask, svfloat32_t x)
        {
            x = svmax_f32_x(mask, svmin_f32_x(mask, x, svdup_n_f32(126.99999f)), svdup_n_f32(-126.99999f));
            svint32_t ipart = svcvt_s32_f32_x(mask, svsub_n_f32_x(mask, x, 0.5f));
            svfloat32_t fpart = svsub_f32_x(mask, x, svcvt_f32_s32_x(mask, ipart));
            svfloat32_t expipart = svreinterpret_f32_s32(svlsl_n_s32_x(mask, svadd_n_s32_x(mask, ipart, 127), 23));
            return svmul_f32_x(mask, expipart, SoftmaxPoly5(mask, fpart));
        }

        SIMD_INLINE svfloat32_t SoftmaxExponent(const svbool_t& mask, svfloat32_t value)
        {
            return SoftmaxExp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
        }

        void SynetSoftmax16b21(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            const size_t F = svcntw(), DF = 2 * F;
            const svbool_t body16 = svwhilelt_b16((size_t)0, F);
            const svbool_t body32 = svptrue_b32();
            size_t o = 0;
            for (; o + F <= outer; o += F)
            {
                svuint16x2_t s = svld2_u16(body16, src);
                svfloat32_t src0 = BFloat16ToFloat32(svget2(s, 0), body32);
                svfloat32_t src1 = BFloat16ToFloat32(svget2(s, 1), body32);
                svfloat32_t max = svmax_f32_x(body32, src0, src1);
                svfloat32_t exp0 = SoftmaxExponent(body32, svsub_f32_x(body32, src0, max));
                svfloat32_t exp1 = SoftmaxExponent(body32, svsub_f32_x(body32, src1, max));
                svfloat32_t sum = svadd_f32_x(body32, exp0, exp1);
                svst2_u16(body16, dst, svcreate2_u16(Float32ToBFloat16Lo(svdiv_f32_x(body32, exp0, sum), body32),
                    Float32ToBFloat16Lo(svdiv_f32_x(body32, exp1, sum), body32)));
                src += DF;
                dst += DF;
            }
            if (o < outer)
            {
                svbool_t mask16 = svwhilelt_b16(o, outer);
                svbool_t mask32 = svwhilelt_b32(o, outer);
                svuint16x2_t s = svld2_u16(mask16, src);
                svfloat32_t src0 = BFloat16ToFloat32(svget2(s, 0), mask32);
                svfloat32_t src1 = BFloat16ToFloat32(svget2(s, 1), mask32);
                svfloat32_t max = svmax_f32_x(mask32, src0, src1);
                svfloat32_t exp0 = SoftmaxExponent(mask32, svsub_f32_x(mask32, src0, max));
                svfloat32_t exp1 = SoftmaxExponent(mask32, svsub_f32_x(mask32, src1, max));
                svfloat32_t sum = svadd_f32_x(mask32, exp0, exp1);
                svst2_u16(mask16, dst, svcreate2_u16(Float32ToBFloat16Lo(svdiv_f32_x(mask32, exp0, sum), mask32),
                    Float32ToBFloat16Lo(svdiv_f32_x(mask32, exp1, sum), mask32)));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftmax16b31(const svbool_t& mask, svfloat32_t buf[3])
        {
            svfloat32_t max = svmax_f32_x(mask, buf[0], svmax_f32_x(mask, buf[1], buf[2]));
            buf[0] = SoftmaxExponent(mask, svsub_f32_x(mask, buf[0], max));
            buf[1] = SoftmaxExponent(mask, svsub_f32_x(mask, buf[1], max));
            buf[2] = SoftmaxExponent(mask, svsub_f32_x(mask, buf[2], max));
            svfloat32_t sum = svadd_f32_x(mask, buf[0], svadd_f32_x(mask, buf[1], buf[2]));
            buf[0] = svdiv_f32_x(mask, buf[0], sum);
            buf[1] = svdiv_f32_x(mask, buf[1], sum);
            buf[2] = svdiv_f32_x(mask, buf[2], sum);
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            const size_t F = svcntw(), DF = 3 * F;
            const svbool_t body16 = svwhilelt_b16((size_t)0, F);
            const svbool_t body32 = svptrue_b32();
            svfloat32_t buf[3];
            size_t o = 0;
            for (; o + F <= outer; o += F)
            {
                svuint16x3_t s = svld3_u16(body16, src);
                buf[0] = BFloat16ToFloat32(svget3(s, 0), body32);
                buf[1] = BFloat16ToFloat32(svget3(s, 1), body32);
                buf[2] = BFloat16ToFloat32(svget3(s, 2), body32);
                SynetSoftmax16b31(body32, buf);
                svst3_u16(body16, dst, svcreate3_u16(Float32ToBFloat16Lo(buf[0], body32),
                    Float32ToBFloat16Lo(buf[1], body32), Float32ToBFloat16Lo(buf[2], body32)));
                src += DF;
                dst += DF;
            }
            if (o < outer)
            {
                svbool_t mask16 = svwhilelt_b16(o, outer);
                svbool_t mask32 = svwhilelt_b32(o, outer);
                svuint16x3_t s = svld3_u16(mask16, src);
                buf[0] = BFloat16ToFloat32(svget3(s, 0), mask32);
                buf[1] = BFloat16ToFloat32(svget3(s, 1), mask32);
                buf[2] = BFloat16ToFloat32(svget3(s, 2), mask32);
                SynetSoftmax16b31(mask32, buf);
                svst3_u16(mask16, dst, svcreate3_u16(Float32ToBFloat16Lo(buf[0], mask32),
                    Float32ToBFloat16Lo(buf[1], mask32), Float32ToBFloat16Lo(buf[2], mask32)));
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetSoftmax16bX1(const uint16_t* src, size_t outer, size_t count, uint16_t* dst)
        {
            const size_t F = svcntw();
            Array32f buf(count * F);
            Array16u tmp(F);
            size_t o = 0;
            for (; o < outer; o += F)
            {
                size_t n = Simd::Min(F, outer - o);
                svbool_t mask = svwhilelt_b32((size_t)0, n);
                svuint32_t offsets = svmul_n_u32_x(mask, svindex_u32(0, 1), (uint32_t)count);
                svfloat32_t max = svdup_n_f32(-FLT_MAX);
                for (size_t c = 0; c < count; ++c)
                {
                    svfloat32_t value = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + c, offsets), mask);
                    max = svmax_f32_x(mask, max, value);
                    svst1_f32(mask, buf.data + c * F, value);
                }
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t c = 0; c < count; ++c)
                {
                    svfloat32_t value = SoftmaxExponent(mask, svsub_f32_x(mask, svld1_f32(mask, buf.data + c * F), max));
                    sum = svadd_f32_x(mask, sum, value);
                    svst1_f32(mask, buf.data + c * F, value);
                }
                for (size_t c = 0; c < count; ++c)
                {
                    svst1h_u32(mask, tmp.data, Float32ToBFloat16(svdiv_f32_x(mask, svld1_f32(mask, buf.data + c * F), sum), mask));
                    for (size_t i = 0; i < n; ++i)
                        dst[i * count + c] = tmp[i];
                }
                src += count * n;
                dst += count * n;
            }
        }

        void SynetSoftmax16bInner(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
        {
            const size_t F = svcntw();
            Array32f _buf(inner * (count + 2));
            float* max = _buf.data, * sum = _buf.data + inner, * buf = sum + inner, * b;
            for (size_t o = 0; o < outer; ++o)
            {
                BFloat16ToFloat32(src, count * inner, buf);
                memcpy(max, buf, inner * sizeof(float));
                b = buf + inner;
                for (size_t c = 1; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; i += F)
                    {
                        svbool_t mask = svwhilelt_b32(i, inner);
                        svst1_f32(mask, max + i, svmax_f32_x(mask, svld1_f32(mask, b + i), svld1_f32(mask, max + i)));
                    }
                    b += inner;
                }

                b = buf;
                memset(sum, 0, inner * sizeof(float));
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; i += F)
                    {
                        svbool_t mask = svwhilelt_b32(i, inner);
                        svfloat32_t _d = SoftmaxExponent(mask, svsub_f32_x(mask, svld1_f32(mask, b + i), svld1_f32(mask, max + i)));
                        svst1_f32(mask, b + i, _d);
                        svst1_f32(mask, sum + i, svadd_f32_x(mask, _d, svld1_f32(mask, sum + i)));
                    }
                    b += inner;
                }

                b = buf;
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; i += F)
                    {
                        svbool_t mask = svwhilelt_b32(i, inner);
                        svst1h_u32(mask, dst + i, Float32ToBFloat16(svdiv_f32_x(mask, svld1_f32(mask, b + i), svld1_f32(mask, sum + i)), mask));
                    }
                    b += inner;
                    dst += inner;
                }
                src += count * inner;
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
                SynetSoftmax16bInner(src, outer, count, inner, dst);
        }
    }
#endif
}
