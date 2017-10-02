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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i MulDiv32(__m256i dividend, __m256i divisor, const __m256 & KF_255_DIV_6)
        {
            return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_mul_ps(KF_255_DIV_6, _mm256_cvtepi32_ps(dividend)), _mm256_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m256i MulDiv16(__m256i dividend, __m256i divisor, const __m256 & KF_255_DIV_6)
        {
            const __m256i quotientLo = MulDiv32(_mm256_unpacklo_epi16(dividend, K_ZERO), _mm256_unpacklo_epi16(divisor, K_ZERO), KF_255_DIV_6);
            const __m256i quotientHi = MulDiv32(_mm256_unpackhi_epi16(dividend, K_ZERO), _mm256_unpackhi_epi16(divisor, K_ZERO), KF_255_DIV_6);
            return _mm256_packs_epi32(quotientLo, quotientHi);
        }

        SIMD_INLINE __m256i AdjustedYuvToHue16(__m256i y, __m256i u, __m256i v, const __m256 & KF_255_DIV_6)
        {
            const __m256i red = AdjustedYuvToRed16(y, v);
            const __m256i green = AdjustedYuvToGreen16(y, u, v);
            const __m256i blue = AdjustedYuvToBlue16(y, u);
            const __m256i max = MaxI16(red, green, blue);
            const __m256i range = _mm256_subs_epi16(max, MinI16(red, green, blue));

            const __m256i redMaxMask = _mm256_cmpeq_epi16(red, max);
            const __m256i greenMaxMask = _mm256_andnot_si256(redMaxMask, _mm256_cmpeq_epi16(green, max));
            const __m256i blueMaxMask = _mm256_andnot_si256(redMaxMask, _mm256_andnot_si256(greenMaxMask, K_INV_ZERO));

            const __m256i redMaxCase = _mm256_and_si256(redMaxMask,
                _mm256_add_epi16(_mm256_sub_epi16(green, blue), _mm256_mullo_epi16(range, K16_0006)));
            const __m256i greenMaxCase = _mm256_and_si256(greenMaxMask,
                _mm256_add_epi16(_mm256_sub_epi16(blue, red), _mm256_mullo_epi16(range, K16_0002)));
            const __m256i blueMaxCase = _mm256_and_si256(blueMaxMask,
                _mm256_add_epi16(_mm256_sub_epi16(red, green), _mm256_mullo_epi16(range, K16_0004)));

            const __m256i dividend = _mm256_or_si256(_mm256_or_si256(redMaxCase, greenMaxCase), blueMaxCase);

            return _mm256_andnot_si256(_mm256_cmpeq_epi16(range, K_ZERO), _mm256_and_si256(MulDiv16(dividend, range, KF_255_DIV_6), K16_00FF));
        }

        SIMD_INLINE __m256i YuvToHue16(__m256i y, __m256i u, __m256i v, const __m256 & KF_255_DIV_6)
        {
            return AdjustedYuvToHue16(AdjustY16(y), AdjustUV16(u), AdjustUV16(v), KF_255_DIV_6);
        }

        SIMD_INLINE __m256i YuvToHue8(__m256i y, __m256i u, __m256i v, const __m256 & KF_255_DIV_6)
        {
            return _mm256_packus_epi16(
                YuvToHue16(_mm256_unpacklo_epi8(y, K_ZERO), _mm256_unpacklo_epi8(u, K_ZERO), _mm256_unpacklo_epi8(v, K_ZERO), KF_255_DIV_6),
                YuvToHue16(_mm256_unpackhi_epi8(y, K_ZERO), _mm256_unpackhi_epi8(u, K_ZERO), _mm256_unpackhi_epi8(v, K_ZERO), KF_255_DIV_6));
        }

        template <bool align> SIMD_INLINE void Yuv420pToHue(const uint8_t * y, __m256i u, __m256i v, uint8_t * hue, const __m256 & KF_255_DIV_6)
        {
            Store<align>((__m256i*)(hue), YuvToHue8(Load<align>((__m256i*)(y)),
                _mm256_unpacklo_epi8(u, u), _mm256_unpacklo_epi8(v, v), KF_255_DIV_6));
            Store<align>((__m256i*)(hue + A), YuvToHue8(Load<align>((__m256i*)(y + A)),
                _mm256_unpackhi_epi8(u, u), _mm256_unpackhi_epi8(v, v), KF_255_DIV_6));
        }

        template <bool align> void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride));
            }

            const __m256 KF_255_DIV_6 = _mm256_set1_ps(Base::KF_255_DIV_6);

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, col_hue = 0; colY < bodyWidth; colY += DA, colUV += A, col_hue += DA)
                {
                    __m256i u_ = LoadPermuted<align>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<align>((__m256i*)(v + colUV));
                    Yuv420pToHue<align>(y + colY, u_, v_, hue + col_hue, KF_255_DIV_6);
                    Yuv420pToHue<align>(y + yStride + colY, u_, v_, hue + hueStride + col_hue, KF_255_DIV_6);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuv420pToHue<false>(y + offset, u_, v_, hue + offset, KF_255_DIV_6);
                    Yuv420pToHue<false>(y + yStride + offset, u_, v_, hue + hueStride + offset, KF_255_DIV_6);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                hue += 2 * hueStride;
            }
        }

        template <bool align> void Yuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride));
            }

            const __m256 KF_255_DIV_6 = _mm256_set1_ps(Base::KF_255_DIV_6);

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    Store<align>((__m256i*)(hue + col), YuvToHue8(Load<align>((__m256i*)(y + col)),
                        Load<align>((__m256i*)(u + col)), Load<align>((__m256i*)(v + col)), KF_255_DIV_6));
                }
                if (tail)
                {
                    size_t offset = width - A;
                    Store<false>((__m256i*)(hue + offset), YuvToHue8(Load<false>((__m256i*)(y + offset)),
                        Load<false>((__m256i*)(u + offset)), Load<false>((__m256i*)(v + offset)), KF_255_DIV_6));
                }
                y += yStride;
                u += uStride;
                v += vStride;
                hue += hueStride;
            }
        }

        void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv420pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv420pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }

        void Yuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv444pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv444pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
