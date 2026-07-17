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
            const size_t F = svcntw();
            Array16u tmp0(F), tmp1(F);
            size_t o = 0;
            for (; o < outer; o += F)
            {
                size_t n = Simd::Min(F, outer - o);
                svbool_t mask = svwhilelt_b32((size_t)0, n);
                svuint32_t offsets = svmul_n_u32_x(mask, svindex_u32(0, 1), 2);
                svfloat32_t src0 = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + 0, offsets), mask);
                svfloat32_t src1 = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + 1, offsets), mask);
                svfloat32_t max = svmax_f32_x(mask, src0, src1);
                svfloat32_t exp0 = SoftmaxExponent(mask, svsub_f32_x(mask, src0, max));
                svfloat32_t exp1 = SoftmaxExponent(mask, svsub_f32_x(mask, src1, max));
                svfloat32_t sum = svadd_f32_x(mask, exp0, exp1);
                svst1h_u32(mask, tmp0.data, Float32ToBFloat16(svdiv_f32_x(mask, exp0, sum), mask));
                svst1h_u32(mask, tmp1.data, Float32ToBFloat16(svdiv_f32_x(mask, exp1, sum), mask));
                for (size_t i = 0; i < n; ++i)
                {
                    dst[i * 2 + 0] = tmp0[i];
                    dst[i * 2 + 1] = tmp1[i];
                }
                src += 2 * n;
                dst += 2 * n;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftmax16b31(const svbool_t& mask, svfloat32_t& buf0, svfloat32_t& buf1, svfloat32_t& buf2)
        {
            svfloat32_t max = svmax_f32_x(mask, buf0, svmax_f32_x(mask, buf1, buf2));
            buf0 = SoftmaxExponent(mask, svsub_f32_x(mask, buf0, max));
            buf1 = SoftmaxExponent(mask, svsub_f32_x(mask, buf1, max));
            buf2 = SoftmaxExponent(mask, svsub_f32_x(mask, buf2, max));
            svfloat32_t sum = svadd_f32_x(mask, buf0, svadd_f32_x(mask, buf1, buf2));
            buf0 = svdiv_f32_x(mask, buf0, sum);
            buf1 = svdiv_f32_x(mask, buf1, sum);
            buf2 = svdiv_f32_x(mask, buf2, sum);
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            const size_t F = svcntw();
            Array16u tmp0(F), tmp1(F), tmp2(F);
            size_t o = 0;
            for (; o < outer; o += F)
            {
                size_t n = Simd::Min(F, outer - o);
                svbool_t mask = svwhilelt_b32((size_t)0, n);
                svuint32_t offsets = svmul_n_u32_x(mask, svindex_u32(0, 1), 3);
                svfloat32_t buf0 = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + 0, offsets), mask);
                svfloat32_t buf1 = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + 1, offsets), mask);
                svfloat32_t buf2 = BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src + 2, offsets), mask);
                SynetSoftmax16b31(mask, buf0, buf1, buf2);
                svst1h_u32(mask, tmp0.data, Float32ToBFloat16(buf0, mask));
                svst1h_u32(mask, tmp1.data, Float32ToBFloat16(buf1, mask));
                svst1h_u32(mask, tmp2.data, Float32ToBFloat16(buf2, mask));
                for (size_t i = 0; i < n; ++i)
                {
                    dst[i * 3 + 0] = tmp0[i];
                    dst[i * 3 + 1] = tmp1[i];
                    dst[i * 3 + 2] = tmp2[i];
                }
                src += 3 * n;
                dst += 3 * n;
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
