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
#include "Simd/SimdMemory.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdGather.h"
#include "Simd/SimdPow.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void SynetSoftmaxLayerForward21(const float* src, size_t outer, float* dst)
        {
            Exp exp;
            size_t aligned = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < aligned; o += F)
            {
                __m128 s0 = _mm_loadu_ps(src + 0);
                __m128 s1 = _mm_loadu_ps(src + F);
                __m128 ss0 = _mm_shuffle_ps(s0, s1, 0x88);
                __m128 ss1 = _mm_shuffle_ps(s0, s1, 0xDD);
                __m128 max = _mm_max_ps(ss0, ss1);
                __m128 exp0 = exp.Exponent(_mm_sub_ps(ss0, max));
                __m128 exp1 = exp.Exponent(_mm_sub_ps(ss1, max));
                __m128 sum = _mm_add_ps(exp0, exp1);
                __m128 d0 = _mm_div_ps(exp0, sum);
                __m128 d1 = _mm_div_ps(exp1, sum);
                _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(d0, d1));
                _mm_storeu_ps(dst + F, _mm_unpackhi_ps(d0, d1));
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

        SIMD_INLINE void SynetSoftmaxLayerForward31(const Exp& exp, __m128 buf[3])
        {
            __m128 max = _mm_max_ps(buf[0], _mm_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm_sub_ps(buf[2], max));
            __m128 sum = _mm_add_ps(buf[0], _mm_add_ps(buf[1], buf[2]));
            buf[0] = _mm_div_ps(buf[0], sum);
            buf[1] = _mm_div_ps(buf[1], sum);
            buf[2] = _mm_div_ps(buf[2], sum);
        }

        void SynetSoftmaxLayerForward31(const float* src, size_t outer, float* dst)
        {
            Exp exp;
            __m128 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                buf[0] = Gather<3>(src + 0);
                buf[1] = Gather<3>(src + 1);
                buf[2] = Gather<3>(src + 2);
                SynetSoftmaxLayerForward31(exp, buf);
                Scater<3>(dst + 0, buf[0]);
                Scater<3>(dst + 1, buf[1]);
                Scater<3>(dst + 2, buf[2]);
                src += 3 * F;
                dst += 3 * F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                buf[0] = Gather<3>(src + 0, tail);
                buf[1] = Gather<3>(src + 1, tail);
                buf[2] = Gather<3>(src + 2, tail);
                SynetSoftmaxLayerForward31(exp, buf);
                Scater<3>(dst + 0, buf[0], tail);
                Scater<3>(dst + 1, buf[1], tail);
                Scater<3>(dst + 2, buf[2], tail);
            }
        }

        SIMD_INLINE void LoadTansp4x4(const float* src, size_t count, float* dst, __m128& max)
        {
            __m128 a0 = _mm_loadu_ps(src + 0 * count);
            __m128 a1 = _mm_loadu_ps(src + 1 * count);
            __m128 a2 = _mm_loadu_ps(src + 2 * count);
            __m128 a3 = _mm_loadu_ps(src + 3 * count);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpacklo_ps(a1, a3);
            __m128 b2 = _mm_unpackhi_ps(a0, a2);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            a0 = _mm_unpacklo_ps(b0, b1);
            max = _mm_max_ps(max, a0);
            _mm_storeu_ps(dst + 0 * F, a0);
            a1 = _mm_unpackhi_ps(b0, b1);
            max = _mm_max_ps(max, a1);
            _mm_storeu_ps(dst + 1 * F, a1);
            a2 = _mm_unpacklo_ps(b2, b3);
            max = _mm_max_ps(max, a2);
            _mm_storeu_ps(dst + 2 * F, a2);
            a3 = _mm_unpackhi_ps(b2, b3);
            max = _mm_max_ps(max, a3);
            _mm_storeu_ps(dst + 3 * F, a3);
        }

        SIMD_INLINE void StoreTansp4x4(const float* src, __m128 k, float* dst, size_t count)
        {
            __m128 a0 = _mm_mul_ps(_mm_loadu_ps(src + 0 * F), k);
            __m128 a1 = _mm_mul_ps(_mm_loadu_ps(src + 1 * F), k);
            __m128 a2 = _mm_mul_ps(_mm_loadu_ps(src + 2 * F), k);
            __m128 a3 = _mm_mul_ps(_mm_loadu_ps(src + 3 * F), k);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpacklo_ps(a1, a3);
            __m128 b2 = _mm_unpackhi_ps(a0, a2);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            _mm_storeu_ps(dst + 0 * count, _mm_unpacklo_ps(b0, b1));
            _mm_storeu_ps(dst + 1 * count, _mm_unpackhi_ps(b0, b1));
            _mm_storeu_ps(dst + 2 * count, _mm_unpacklo_ps(b2, b3));
            _mm_storeu_ps(dst + 3 * count, _mm_unpackhi_ps(b2, b3));
        }

        void SynetSoftmaxLayerForwardX1(const float* src, size_t outer, size_t count, float* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                __m128 _max = _mm_setzero_ps();
                for (c = 0; c < countF; c += F)
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                if (c < count)
                {
                    c = count - F;
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                }
                __m128 _sum = _mm_setzero_ps();
                for (size_t c = 0; c < count; ++c)
                {
                    __m128 _exp = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm_add_ps(_sum, _exp);
                    _mm_storeu_ps(buf.data + c * F, _exp);
                }
                __m128 _k = _mm_div_ps(_mm_set1_ps(1.0f), _sum);
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

        void SynetSoftmaxLayerForward(const float* src, size_t outer, size_t count, size_t inner, float* dst)
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
                Exp exp;
                size_t aligned = Simd::AlignLo(inner, F);
                Array32f tmp(inner * 2);
                const float* s;
                float* max = tmp.data, * sum = tmp.data + inner, * d;
                for (size_t o = 0; o < outer; ++o)
                {
                    memcpy(max, src, inner * sizeof(float));
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            _mm_storeu_ps(max + i, _mm_max_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
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
                            __m128 _d = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
                            _mm_storeu_ps(d + i, _d);
                            _mm_storeu_ps(sum + i, _mm_add_ps(_d, _mm_loadu_ps(sum + i)));
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
                            _mm_storeu_ps(d + i, _mm_div_ps(_mm_loadu_ps(d + i), _mm_loadu_ps(sum + i)));
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
