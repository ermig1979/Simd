/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            size_t aligned = AlignLo(size, F);
            Array32f sum(size, true), zero(size, true);

            for (size_t i = 0; i < half; ++i)
            {
                const float * pos = src + i * size;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m128 _pos = Sse::Load<align>(pos + j);
                    Sse::Store<true>(sum.data + j, _mm_add_ps(Sse::Load<true>(sum.data + j), _mm_mul_ps(_pos, _pos)));
                }
                for (; j < size; ++j)
                    sum[j] += Simd::Square(pos[j]);
            }

            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Sse2::Pow pow;
            for (size_t i = 0; i < count; ++i)
            {
                const float * pos = (i < count - half) ? src + half * size : zero.data;
                const float * neg = (i > half) ? src - (half + 1) * size : zero.data;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m128 _pos = Sse::Load<align>(pos + j);
                    __m128 _neg = Sse::Load<align>(neg + j);
                    __m128 _sum = Sse::Load<true>(sum.data + j);
                    _sum = _mm_add_ps(_sum, _mm_sub_ps(_mm_mul_ps(_pos, _pos), _mm_mul_ps(_neg, _neg)));
                    __m128 _src = Sse::Load<align>(src + j);
                    Sse::Store<true>(sum.data + j, _sum);
                    Sse::Store<align>(dst + j, _mm_mul_ps(_src, pow(_mm_add_ps(k0, _mm_mul_ps(k1, _sum)), k2)));
                }
                for (; j < size; ++j)
                {
                    sum[j] += Simd::Square(pos[j]);
                    sum[j] -= Simd::Square(neg[j]);
                    dst[j] = src[j] * Base::Pow(k[0] + k[1] * sum[j], k[2]);
                }
                src += size;
                dst += size;
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(size))
                SynetLrnLayerCrossChannels<true>(src, half, count, size, k, dst);
            else
                SynetLrnLayerCrossChannels<false>(src, half, count, size, k, dst);
        }

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            Sse2::Exp exp;
            if (inner == 1 && count == 2)
            {
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
            else
            {
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
#endif// SIMD_SSE2_ENABLE
}
