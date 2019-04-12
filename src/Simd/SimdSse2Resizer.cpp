/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Base::ResizerByteBilinear(param)
        {
        }

        void ResizerByteBilinear::EstimateIndexAlphaX(size_t srcW, size_t dstW, int32_t * index, int16_t * alpha)
        {
            float scale = (float)srcW / dstW;
            for (size_t dx = 0; dx < dstW; ++dx)
            {
                float a = (float)((dx + 0.5)*scale - 0.5);
                ptrdiff_t i = (ptrdiff_t)::floor(a);
                a -= i;
                if (i < 0)
                {
                    i = 0;
                    a = 0;
                }
                if (i > (ptrdiff_t)srcW - 2)
                {
                    i = srcW - 2;
                    a = 1;
                }
                index[dx] = (int32_t)i;
                alpha[1] = (int16_t)(a * Base::FRACTION_RANGE + 0.5);
                alpha[0] = (int16_t)(Base::FRACTION_RANGE - alpha[1]);
                alpha += 2;
            }
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
            if (_ax.data == 0)
            {
                _ix.Resize(_param.dstW);
                _ax.Resize(_param.dstW*2);
                EstimateIndexAlphaX(_param.srcW, _param.dstW, _ix.data, _ax.data);
                size_t size = AlignHi(_param.dstW, A)*_param.channels * 2;
                _bx[0].Resize(size);
                _bx[1].Resize(size);
            }
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }        
        }

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(__m128i));
            if (type == SimdResizeChannelByte && method == SimdResizeMethodBilinear && (channels == 1 || channels == 2) && dstX >= A)
                return new ResizerByteBilinear(param);
            else
                return Sse::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif//SIMD_SSE2_ENABLE
}

