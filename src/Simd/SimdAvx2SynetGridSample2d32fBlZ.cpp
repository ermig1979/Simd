/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template <int align> SIMD_INLINE float Denormalize32f(float pos, int dim)
        {
            if (align)
                return float((pos + 1) / 2.0f * (dim - 1));
            else
                return float(((pos + 1) * dim - 1) / 2.0f);
        }

        template<int align, int range>  void IndexCoeffs32fBlZ(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float* dy, float* dx, int& yMin, int& yMax)
        {
            size_t dstS4 = AlignLo(dstS, 4), dstS8 = AlignLo(dstS, 8), d = 0;
            const __m256 a = SetFloat((srcW - align) / 2.0f, (srcH - align) / 2.0f);
            const __m256 b = SetFloat((srcW - 1) / 2.0f, (srcH - 1) / 2.0f);
            const __m256i _0 = _mm256_setzero_si256();
            const __m256i _2 = _mm256_set1_epi32(2);
            const __m256i _srcH = _mm256_set1_epi32(srcH + 2);
            const __m256i _srcW = _mm256_set1_epi32(srcW + 2);
            const __m256i _padW = _mm256_set1_epi32(padW);
            __m256i _yMin, _yMax;
            if (range)
            {
                _yMin = _mm256_set1_epi32(yMin);
                _yMax = _mm256_set1_epi32(yMax);
            }
            for (; d < dstS8; d += 8)
            {
                __m256 xy0 = _mm256_fmadd_ps(Load<false>(grd + 0, grd + 8), a, b);
                __m256 xy1 = _mm256_fmadd_ps(Load<false>(grd + 4, grd + 12), a, b);
                __m256 x = _mm256_shuffle_ps(xy0, xy1, 0x88);
                __m256 y = _mm256_shuffle_ps(xy0, xy1, 0xDD);
                __m256 xf = _mm256_round_ps(x, _MM_FROUND_FLOOR);
                __m256 yf = _mm256_round_ps(y, _MM_FROUND_FLOOR);
                _mm256_storeu_ps(dy + d, _mm256_sub_ps(y, yf));
                _mm256_storeu_ps(dx + d, _mm256_sub_ps(x, xf));
                __m256i xi = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(xf), _2), _0), _srcW);
                __m256i yi = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(yf), _2), _0), _srcH);
                _mm256_storeu_si256((__m256i*)(idx + d), _mm256_add_epi32(_mm256_mullo_epi32(_padW, yi), xi));
                if (range)
                {
                    _yMin = _mm256_min_epi32(_yMin, yi);
                    _yMax = _mm256_max_epi32(_yMax, yi);
                }
                grd += 2 * 8;
            }
            for (; d < dstS4; d += 4)
            {
                __m128 xy0 = _mm_fmadd_ps(_mm_loadu_ps(grd + 0), _mm256_castps256_ps128(a), _mm256_castps256_ps128(b));
                __m128 xy1 = _mm_fmadd_ps(_mm_loadu_ps(grd + 4), _mm256_castps256_ps128(a), _mm256_castps256_ps128(b));
                __m128 x = _mm_shuffle_ps(xy0, xy1, 0x88);
                __m128 y = _mm_shuffle_ps(xy0, xy1, 0xDD);
                __m128 xf = _mm_round_ps(x, _MM_FROUND_FLOOR);
                __m128 yf = _mm_round_ps(y, _MM_FROUND_FLOOR);
                _mm_storeu_ps(dy + d, _mm_sub_ps(y, yf));
                _mm_storeu_ps(dx + d, _mm_sub_ps(x, xf));
                __m128i xi = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_cvtps_epi32(xf), _mm256_castsi256_si128(_2)), _mm256_castsi256_si128(_0)), _mm256_castsi256_si128(_srcW));
                __m128i yi = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(_mm_cvtps_epi32(yf), _mm256_castsi256_si128(_2)), _mm256_castsi256_si128(_0)), _mm256_castsi256_si128(_srcH));
                _mm_storeu_si128((__m128i*)(idx + d), _mm_add_epi32(_mm_mullo_epi32(_mm256_castsi256_si128(_padW), yi), xi));
                if (range)
                {
                    _yMin = _mm256_min_epi32(_yMin, _mm256_castsi128_si256(yi));
                    _yMax = _mm256_max_epi32(_yMax, _mm256_castsi128_si256(yi));
                }
                grd += 2 * 4;
            }
            if (range)
            {
                yMin = MinVal32i(_yMin);
                yMax = MaxVal32i(_yMax);
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
                if (range)
                {
                    yMin = Min(yMin, y0);
                    yMax = Max(yMax, y0);
                }
                grd += 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void BilinearInterp32fBlZ(const float* pad0, size_t dstS, int padW, uint32_t* idx, float* dy, float* dx, float* dst)
        {
            size_t dstS4 = AlignLo(dstS, 4), dstS8 = AlignLo(dstS, 8), d = 0;
            const float* pad1 = pad0 + padW;
            __m256 _1 = _mm256_set1_ps(1.0f);
            for (; d < dstS8; d += 8)
            {
                int i0 = idx[d + 0], i1 = idx[d + 1], i2 = idx[d + 2], i3 = idx[d + 3];
                int i4 = idx[d + 4], i5 = idx[d + 5], i6 = idx[d + 6], i7 = idx[d + 7];
                __m256 p0 = Load(pad0 + i0, pad0 + i1, pad0 + i4, pad0 + i5);
                __m256 p1 = Load(pad0 + i2, pad0 + i3, pad0 + i6, pad0 + i7);
                __m256 p00 = _mm256_shuffle_ps(p0, p1, 0x88);
                __m256 p01 = _mm256_shuffle_ps(p0, p1, 0xDD);
                p0 = Load(pad1 + i0, pad1 + i1, pad1 + i4, pad1 + i5);
                p1 = Load(pad1 + i2, pad1 + i3, pad1 + i6, pad1 + i7);
                __m256 p10 = _mm256_shuffle_ps(p0, p1, 0x88);
                __m256 p11 = _mm256_shuffle_ps(p0, p1, 0xDD);
                __m256 dy1 = _mm256_loadu_ps(dy + d);
                __m256 dy0 = _mm256_sub_ps(_1, dy1);
                __m256 dx1 = _mm256_loadu_ps(dx + d);
                __m256 dx0 = _mm256_sub_ps(_1, dx1);
                __m256 d0 = _mm256_fmadd_ps(dx0, p00, _mm256_mul_ps(dx1, p01));
                __m256 d1 = _mm256_fmadd_ps(dx0, p10, _mm256_mul_ps(dx1, p11));
                _mm256_storeu_ps(dst + d, _mm256_fmadd_ps(dy0, d0, _mm256_mul_ps(dy1, d1)));
            }
            for (; d < dstS4; d += 4)
            {
                int i0 = idx[d + 0], i1 = idx[d + 1], i2 = idx[d + 2], i3 = idx[d + 3];
                __m128 p0 = Sse41::Load(pad0 + i0, pad0 + i1);
                __m128 p1 = Sse41::Load(pad0 + i2, pad0 + i3);
                __m128 p00 = _mm_shuffle_ps(p0, p1, 0x88);
                __m128 p01 = _mm_shuffle_ps(p0, p1, 0xDD);
                p0 = Sse41::Load(pad1 + i0, pad1 + i1);
                p1 = Sse41::Load(pad1 + i2, pad1 + i3);
                __m128 p10 = _mm_shuffle_ps(p0, p1, 0x88);
                __m128 p11 = _mm_shuffle_ps(p0, p1, 0xDD);
                __m128 dy1 = _mm_loadu_ps(dy + d);
                __m128 dy0 = _mm_sub_ps(_mm256_castps256_ps128(_1), dy1);
                __m128 dx1 = _mm_loadu_ps(dx + d);
                __m128 dx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), dx1);
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
            : Sse41::SynetGridSample2d32fBlZ(param)
        {
            if (_sparse)
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 1> : IndexCoeffs32fBlZ<0, 1>;
            else
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 0> : IndexCoeffs32fBlZ<0, 0>;
            _bilinearInterp = BilinearInterp32fBlZ;
        }
    }
#endif
}
