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

#include "Simd/SimdSynetGridSample.h"

#include "Simd/SimdLoad.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Sse41
    {
        template <SimdBool align> SIMD_INLINE float Denormalize32f(float pos, int dim)
        {
            if (align)
                return float((pos + 1) / 2.0f * (dim - 1));
            else
                return float(((pos + 1) * dim - 1) / 2.0f);
        }

        template<SimdBool align>  void IndexCoeffs32fBlZ(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float* dy, float* dx)
        {
            size_t dstSF = AlignLo(dstS, F), d = 0;
            const __m128 a = SetFloat((srcW - align) / 2.0f, (srcH - align) / 2.0f);
            const __m128 b = SetFloat((srcW - 1) / 2.0f, (srcH - 1) / 2.0f);
            const __m128i _0 = _mm_setzero_si128();
            const __m128i _2 = _mm_set1_epi32(2);
            const __m128i _srcH = _mm_set1_epi32(srcH + 2);
            const __m128i _srcW = _mm_set1_epi32(srcW + 2);
            const __m128i _padW = _mm_set1_epi32(padW);
            for (; d < dstSF; d += F)
            {
                __m128 xy0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(grd + 0), a), b);
                __m128 xy1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(grd + F), a), b);
                __m128 x = _mm_shuffle_ps(xy0, xy1, 0x88);
                __m128 y = _mm_shuffle_ps(xy0, xy1, 0xDD);
                __m128 xf = _mm_round_ps(x, _MM_FROUND_FLOOR);
                __m128 yf = _mm_round_ps(y, _MM_FROUND_FLOOR);
                _mm_storeu_ps(dy + d, _mm_sub_ps(y, yf));
                _mm_storeu_ps(dx + d, _mm_sub_ps(x, xf));
                __m128i xi = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_cvtps_epi32(xf), _2), _0), _srcW);
                __m128i yi = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_cvtps_epi32(yf), _2), _0), _srcH);
                _mm_storeu_si128((__m128i*)(idx + d), _mm_add_epi32(_mm_mullo_epi32(_padW, yi), xi));
                grd += 2 * F;
            }
            for (; d < dstS; ++d)
            {
                float x = Denormalize32f<align>(grd[0], srcW);
                float y = Denormalize32f<align>(grd[1], srcH);
                int x0 = int(std::floor(x));
                int y0 = int(std::floor(y));
                dy[d] = y - float(y0);
                dx[d] = x - float(x0);
                x0 = Simd::RestrictRange(x0, -2, srcW) + 2;
                y0 = Simd::RestrictRange(y0, -2, srcH) + 2;
                idx[d] = padW * y0 + x0;
                grd += 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void BilinearInterp32fBlZ(const float* pad0, size_t dstS, int padW, uint32_t* idx, float* dy, float* dx, float* dst)
        {
            size_t dstSF = AlignLo(dstS, F), d = 0;
            const float* pad1 = pad0 + padW;
            __m128 p0, p1, _1 = _mm_set1_ps(1.0f);
            for (; d < dstSF; d += F)
            {
                int i0 = idx[d + 0], i1 = idx[d + 1], i2 = idx[d + 2], i3 = idx[d + 3];
                p0 = Load(pad0 + i0, pad0 + i1);
                p1 = Load(pad0 + i2, pad0 + i3);
                __m128 p00 = _mm_shuffle_ps(p0, p1, 0x88);
                __m128 p01 = _mm_shuffle_ps(p0, p1, 0xDD);
                p0 = Load(pad1 + i0, pad1 + i1);
                p1 = Load(pad1 + i2, pad1 + i3);
                __m128 p10 = _mm_shuffle_ps(p0, p1, 0x88);
                __m128 p11 = _mm_shuffle_ps(p0, p1, 0xDD);
                __m128 dy1 = _mm_loadu_ps(dy + d);
                __m128 dy0 = _mm_sub_ps(_1, dy1);
                __m128 dx1 = _mm_loadu_ps(dx + d);
                __m128 dx0 = _mm_sub_ps(_1, dx1);
                __m128 d0 = _mm_add_ps(_mm_mul_ps(dx0, p00), _mm_mul_ps(dx1, p01));
                __m128 d1 = _mm_add_ps(_mm_mul_ps(dx0, p10), _mm_mul_ps(dx1, p11));
                _mm_storeu_ps(dst + d, _mm_add_ps(_mm_mul_ps(dy0, d0), _mm_mul_ps(dy1, d1)));
            }
            for (; d < dstS; ++d)
            {
                int offs = idx[d];
                float p00 = pad0[offs + 0];
                float p01 = pad0[offs + 1];
                float p10 = pad1[offs + 0];
                float p11 = pad1[offs + 1];
                float dy1 = dy[d];
                float dy0 = 1.0f - dy1;
                float dx1 = dx[d];
                float dx0 = 1.0f - dx1;
                dst[d] = dy0 * (dx0 * p00 + dx1 * p01) + dy1 * (dx0 * p10 + dx1 * p11);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGridSample2d32fBlZ::SynetGridSample2d32fBlZ(const GridSample2dParam& param)
            : Base::SynetGridSample2d32fBlZ(param)
        {
            _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<SimdTrue> : IndexCoeffs32fBlZ<SimdFalse>;
            _bilinearInterp = BilinearInterp32fBlZ;
        }
    }
#endif
}
