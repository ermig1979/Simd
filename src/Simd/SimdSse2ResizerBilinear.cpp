/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdResizer.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Base::ResizerByteBilinear(param)
        {
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            _ix.Resize(_param.dstW);
            _ax.Resize(AlignHi(_param.dstW, A) * 2);
            float scale = (float)_param.srcW / _param.dstW;
            for (size_t dx = 0; dx < _param.dstW; ++dx)
            {
                float a = (float)((dx + 0.5)*scale - 0.5);
                ptrdiff_t i = (ptrdiff_t)::floor(a);
                a -= i;
                if (i < 0)
                {
                    i = 0;
                    a = 0;
                }
                if (i > (ptrdiff_t)_param.srcW - 2)
                {
                    i = _param.srcW - 2;
                    a = 1;
                }
                _ix.data[dx] = (int32_t)i;
                _ax.data[2 * dx + 1] = (int16_t)(a * Base::FRACTION_RANGE + 0.5);
                _ax.data[2 * dx + 0] = (int16_t)(Base::FRACTION_RANGE - _ax.data[2 * dx + 1]);
            }
            size_t size = AlignHi(_param.dstW, A)*_param.channels * 2;
            _bx[0].Resize(size);
            _bx[1].Resize(size);
        }

        template <size_t N> void ResizerByteBilinearInterpolateX(const __m128i * alpha, __m128i * buffer);

        SIMD_INLINE void ResizerByteBilinearInterpolateX1(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(src, K_ZERO), _mm_load_si128(alpha + 0));
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(src, K_ZERO), _mm_load_si128(alpha + 1));
            _mm_store_si128(buffer, _mm_packs_epi32(lo, hi));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const __m128i * alpha, __m128i * buffer)
        {
            ResizerByteBilinearInterpolateX1(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX1(alpha + 2, buffer + 1);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i a = _mm_load_si128(alpha);
            __m128i u = _mm_madd_epi16(_mm_and_si128(src, K16_00FF), a);
            __m128i v = _mm_madd_epi16(_mm_and_si128(_mm_srli_si128(src, 1), K16_00FF), a);
            _mm_store_si128(buffer, _mm_or_si128(u, _mm_slli_si128(v, 2)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const __m128i * alpha, __m128i * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + 1, buffer + 1);
        }

        const __m128i K16_FRACTION_ROUND_TERM = SIMD_MM_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m128i ResizerByteBilinearInterpolateY(const __m128i * pbx0, const __m128i * pbx1, __m128i alpha[2])
        {
            __m128i sum = _mm_add_epi16(_mm_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm_srli_epi16(_mm_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m128i alpha[2], uint8_t * dst)
        {
            __m128i lo = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 0, (__m128i*)bx1 + 0, alpha);
            __m128i hi = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 1, (__m128i*)bx1 + 1, alpha);
            Store<false>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(_param.dstW >= A);

            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t stepB = A / N;
            const size_t stepA = DA / N;
            size_t bufW = AlignHi(_param.dstW, stepB);

            ptrdiff_t previous = -2;
            __m128i a[2];
            uint8_t * pbx[2] = { _bx[0].data, _bx[1].data };

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)pbx[k];
                    const One * ps = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < _param.dstW; x++)
                        pb[x] = *(Two *)(ps + _ix[x]);

                    for (size_t ib = 0, ia = 0; ib < bufW; ib += stepB, ia += stepA)
                        ResizerByteBilinearInterpolateX<N>((__m128i*)(_ax.data + ia), (__m128i*)(pb + ib));
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(pbx[0] + ib, pbx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(pbx[0] + i, pbx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            EstimateParams();
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }        
        }

        //---------------------------------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam& param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(rs, F);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float* pb = pbx[k];
                    const float* ps = src + (sy + k) * srcStride;
                    size_t dx = 0;
                    if (cn == 1)
                    {
                        __m128 _1 = _mm_set1_ps(1.0f);
                        for (; dx < rsa; dx += F)
                        {
                            __m128 s01 = Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                            __m128 s23 = Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                            __m128 fx1 = _mm_load_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s01, s23, 0x88));
                            __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s01, s23, 0xDD));
                            _mm_store_ps(pb + dx, _mm_add_ps(m0, m1));
                        }
                    }
                    if (cn == 3 && rs > 3)
                    {
                        __m128 _1 = _mm_set1_ps(1.0f);
                        size_t rs3 = rs - 3;
                        for (; dx < rs3; dx += 3)
                        {
                            __m128 s0 = _mm_loadu_ps(ps + _ix[dx] + 0);
                            __m128 s1 = _mm_loadu_ps(ps + _ix[dx] + 3);
                            __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                    }
                }

                size_t dx = 0;
                __m128 _fy0 = _mm_set1_ps(fy0);
                __m128 _fy1 = _mm_set1_ps(fy1);
                for (; dx < rsa; dx += F)
                {
                    __m128 m0 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx), _fy0);
                    __m128 m1 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx), _fy1);
                    _mm_storeu_ps(dst + dx, _mm_add_ps(m0, m1));
                }
                for (; dx < rs; dx++)
                    dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
            }
        }
    }
#endif//SIMD_SSE2_ENABLE
}

