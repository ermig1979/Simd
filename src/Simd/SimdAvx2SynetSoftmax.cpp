/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGather.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        void SynetSoftmaxLayerForward21(const float* src, size_t outer, float* dst)
        {
            Avx2::Exp exp;
            size_t aligned = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < aligned; o += F)
            {
                __m256 s0 = _mm256_loadu_ps(src + 0);
                __m256 s1 = _mm256_loadu_ps(src + F);
                __m256 ss0 = _mm256_shuffle_ps(s0, s1, 0x88);
                __m256 ss1 = _mm256_shuffle_ps(s0, s1, 0xDD);
                __m256 max = _mm256_max_ps(ss0, ss1);
                __m256 exp0 = exp.Exponent(_mm256_sub_ps(ss0, max));
                __m256 exp1 = exp.Exponent(_mm256_sub_ps(ss1, max));
                __m256 sum = _mm256_add_ps(exp0, exp1);
                __m256 d0 = _mm256_div_ps(exp0, sum);
                __m256 d1 = _mm256_div_ps(exp1, sum);
                _mm256_storeu_ps(dst + 0, _mm256_unpacklo_ps(d0, d1));
                _mm256_storeu_ps(dst + F, _mm256_unpackhi_ps(d0, d1));
                src += DF;
                dst += DF;
            }
            for (; o < outer; ++o)
            {
                float max = Simd::Max(src[0], src[1]);
                float exp0 = ::exp(src[0] - max);
                float exp1 = ::exp(src[1] - max);
                float sum = exp0 + exp1;
                dst[0] = exp0 / sum;
                dst[1] = exp1 / sum;
                src += 2;
                dst += 2;
            }
        }

        SIMD_INLINE void SynetSoftmaxLayerForward31(const Avx2::Exp& exp, __m256 buf[3])
        {
            __m256 max = _mm256_max_ps(buf[0], _mm256_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm256_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm256_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm256_sub_ps(buf[2], max));
            __m256 sum = _mm256_add_ps(buf[0], _mm256_add_ps(buf[1], buf[2]));
            buf[0] = _mm256_div_ps(buf[0], sum);
            buf[1] = _mm256_div_ps(buf[1], sum);
            buf[2] = _mm256_div_ps(buf[2], sum);
        }

        void SynetSoftmaxLayerForward31(const float* src, size_t outer, float* dst)
        {
            Avx2::Exp exp;
            __m256 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                buf[0] = Avx2::Gather<3>(src + 0);
                buf[1] = Avx2::Gather<3>(src + 1);
                buf[2] = Avx2::Gather<3>(src + 2);
                SynetSoftmaxLayerForward31(exp, buf);
                Avx::Scater<3>(dst + 0, buf[0]);
                Avx::Scater<3>(dst + 1, buf[1]);
                Avx::Scater<3>(dst + 2, buf[2]);
                src += 3 * F;
                dst += 3 * F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                buf[0] = Avx::Gather<3>(src + 0, tail);
                buf[1] = Avx::Gather<3>(src + 1, tail);
                buf[2] = Avx::Gather<3>(src + 2, tail);
                SynetSoftmaxLayerForward31(exp, buf);
                Avx::Scater<3>(dst + 0, buf[0], tail);
                Avx::Scater<3>(dst + 1, buf[1], tail);
                Avx::Scater<3>(dst + 2, buf[2], tail);
            }
        }

        SIMD_INLINE void LoadTansp8x8(const float* src, size_t count, float* dst, __m256& max)
        {
            __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm256_loadu_ps(src + 0 * count);
            a1 = _mm256_loadu_ps(src + 1 * count);
            a2 = _mm256_loadu_ps(src + 2 * count);
            a3 = _mm256_loadu_ps(src + 3 * count);
            a4 = _mm256_loadu_ps(src + 4 * count);
            a5 = _mm256_loadu_ps(src + 5 * count);
            a6 = _mm256_loadu_ps(src + 6 * count);
            a7 = _mm256_loadu_ps(src + 7 * count);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);
            b4 = _mm256_unpacklo_ps(a4, a6);
            b5 = _mm256_unpacklo_ps(a5, a7);
            b6 = _mm256_unpackhi_ps(a4, a6);
            b7 = _mm256_unpackhi_ps(a5, a7);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);
            a4 = _mm256_unpacklo_ps(b4, b5);
            a5 = _mm256_unpackhi_ps(b4, b5);
            a6 = _mm256_unpacklo_ps(b6, b7);
            a7 = _mm256_unpackhi_ps(b6, b7);

            b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
            b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
            b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
            b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
            b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
            b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
            b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
            b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

            max = _mm256_max_ps(max, b0);
            max = _mm256_max_ps(max, b1);
            max = _mm256_max_ps(max, b2);
            max = _mm256_max_ps(max, b3);
            max = _mm256_max_ps(max, b4);
            max = _mm256_max_ps(max, b5);
            max = _mm256_max_ps(max, b6);
            max = _mm256_max_ps(max, b7);

            _mm256_storeu_ps(dst + 0 * F, b0);
            _mm256_storeu_ps(dst + 1 * F, b1);
            _mm256_storeu_ps(dst + 2 * F, b2);
            _mm256_storeu_ps(dst + 3 * F, b3);
            _mm256_storeu_ps(dst + 4 * F, b4);
            _mm256_storeu_ps(dst + 5 * F, b5);
            _mm256_storeu_ps(dst + 6 * F, b6);
            _mm256_storeu_ps(dst + 7 * F, b7);
        }

        SIMD_INLINE void StoreTansp8x8(const float* src, __m256 k, float* dst, size_t count)
        {
            __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm256_mul_ps(_mm256_loadu_ps(src + 0 * F), k);
            a1 = _mm256_mul_ps(_mm256_loadu_ps(src + 1 * F), k);
            a2 = _mm256_mul_ps(_mm256_loadu_ps(src + 2 * F), k);
            a3 = _mm256_mul_ps(_mm256_loadu_ps(src + 3 * F), k);
            a4 = _mm256_mul_ps(_mm256_loadu_ps(src + 4 * F), k);
            a5 = _mm256_mul_ps(_mm256_loadu_ps(src + 5 * F), k);
            a6 = _mm256_mul_ps(_mm256_loadu_ps(src + 6 * F), k);
            a7 = _mm256_mul_ps(_mm256_loadu_ps(src + 7 * F), k);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);
            b4 = _mm256_unpacklo_ps(a4, a6);
            b5 = _mm256_unpacklo_ps(a5, a7);
            b6 = _mm256_unpackhi_ps(a4, a6);
            b7 = _mm256_unpackhi_ps(a5, a7);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);
            a4 = _mm256_unpacklo_ps(b4, b5);
            a5 = _mm256_unpackhi_ps(b4, b5);
            a6 = _mm256_unpacklo_ps(b6, b7);
            a7 = _mm256_unpackhi_ps(b6, b7);

            b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
            b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
            b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
            b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
            b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
            b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
            b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
            b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

            _mm256_storeu_ps(dst + 0 * count, b0);
            _mm256_storeu_ps(dst + 1 * count, b1);
            _mm256_storeu_ps(dst + 2 * count, b2);
            _mm256_storeu_ps(dst + 3 * count, b3);
            _mm256_storeu_ps(dst + 4 * count, b4);
            _mm256_storeu_ps(dst + 5 * count, b5);
            _mm256_storeu_ps(dst + 6 * count, b6);
            _mm256_storeu_ps(dst + 7 * count, b7);
        }

        void SynetSoftmaxLayerForwardX1(const float* src, size_t outer, size_t count, float* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                __m256 _max = _mm256_setzero_ps();
                for (c = 0; c < countF; c += F)
                    LoadTansp8x8(src + c, count, buf.data + c * F, _max);
                if (c < count)
                {
                    c = count - F;
                    LoadTansp8x8(src + c, count, buf.data + c * F, _max);
                }
                __m256 _sum = _mm256_setzero_ps();
                for (size_t c = 0; c < count; ++c)
                {
                    __m256 _exp = exp.Exponent(_mm256_sub_ps(_mm256_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm256_add_ps(_sum, _exp);
                    _mm256_storeu_ps(buf.data + c * F, _exp);
                }
                __m256 _k = _mm256_div_ps(_mm256_set1_ps(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp8x8(buf.data + c * F, _k, dst + c, count);
                if (c < count)
                {
                    c = count - F;
                    StoreTansp8x8(buf.data + c * F, _k, dst + c, count);
                }
                src += count * F;
                dst += count * F;
            }
            for (; o < outer; ++o)
            {
                float max = src[0];
                for (size_t c = 1; c < count; ++c)
                    max = Simd::Max(max, src[c]);
                float sum = 0;
                for (size_t c = 0; c < count; ++c)
                {
                    dst[c] = ::exp(src[c] - max);
                    sum += dst[c];
                }
                float k = 1.0f / sum;
                for (size_t c = 0; c < count; ++c)
                    dst[c] *= k;
                src += count;
                dst += count;
            }
        }

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                    SynetSoftmaxLayerForward21(src, outer, dst);
                else if (count == 3)
                    SynetSoftmaxLayerForward31(src, outer, dst);
                else
                    SynetSoftmaxLayerForwardX1(src, outer, count, dst);
            }
            else
            {
                Avx2::Exp exp;
                size_t aligned = Simd::AlignLo(inner, F);
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    memcpy(max, src, inner * sizeof(float));
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            _mm256_storeu_ps(max + i, _mm256_max_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(max + i)));
                        for (; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                        {
                            __m256 _d = exp.Exponent(_mm256_sub_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(max + i)));
                            _mm256_storeu_ps(d + i, _d);
                            _mm256_storeu_ps(sum + i, _mm256_add_ps(_d, _mm256_loadu_ps(sum + i)));
                        }
                        for (; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            _mm256_storeu_ps(d + i, _mm256_div_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(sum + i)));
                        for (; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }
    }
#endif
}
