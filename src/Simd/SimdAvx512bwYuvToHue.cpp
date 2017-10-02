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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i MulDiv32(const __m512i & dividend, const __m512i & divisor, const __m512 & KF_255_DIV_6)
        {
            return _mm512_cvttps_epi32(_mm512_div_ps(_mm512_mul_ps(KF_255_DIV_6, _mm512_cvtepi32_ps(dividend)), _mm512_cvtepi32_ps(divisor)));
        }

        SIMD_INLINE __m512i MulDiv16(const __m512i & dividend, const __m512i & divisor, const __m512 & KF_255_DIV_6)
        {
            const __m512i quotientLo = MulDiv32(_mm512_unpacklo_epi16(dividend, K_ZERO), _mm512_unpacklo_epi16(divisor, K_ZERO), KF_255_DIV_6);
            const __m512i quotientHi = MulDiv32(_mm512_unpackhi_epi16(dividend, K_ZERO), _mm512_unpackhi_epi16(divisor, K_ZERO), KF_255_DIV_6);
            return _mm512_packs_epi32(quotientLo, quotientHi);
        }

        SIMD_INLINE __m512i AdjustedYuvToHue16(const __m512i & y, const __m512i & u, const __m512i & v, const __m512 & KF_255_DIV_6)
        {
            const __m512i red = AdjustedYuvToRed16(y, v);
            const __m512i green = AdjustedYuvToGreen16(y, u, v);
            const __m512i blue = AdjustedYuvToBlue16(y, u);
            const __m512i max = MaxI16(red, green, blue);
            const __m512i range = _mm512_subs_epi16(max, MinI16(red, green, blue));

            const __mmask32 redMaxMask = _mm512_cmpeq_epi16_mask(red, max);
            const __mmask32 greenMaxMask = (~redMaxMask)&_mm512_cmpeq_epi16_mask(green, max);
            const __mmask32 blueMaxMask = ~(redMaxMask | greenMaxMask);

            __m512i dividend = _mm512_maskz_add_epi16(redMaxMask, _mm512_sub_epi16(green, blue), _mm512_mullo_epi16(range, K16_0006));
            dividend = _mm512_mask_add_epi16(dividend, greenMaxMask, _mm512_sub_epi16(blue, red), _mm512_mullo_epi16(range, K16_0002));
            dividend = _mm512_mask_add_epi16(dividend, blueMaxMask, _mm512_sub_epi16(red, green), _mm512_mullo_epi16(range, K16_0004));

            return _mm512_and_si512(MulDiv16(dividend, range, KF_255_DIV_6), _mm512_maskz_set1_epi16(_mm512_cmpneq_epi16_mask(range, K_ZERO), 0xFF));
        }

        template <bool align, bool mask> SIMD_INLINE void YuvToHue(const __m512i & y, const __m512i & u, const __m512i & v, const __m512 & KF_255_DIV_6, uint8_t * hue, __mmask64 tail)
        {
            __m512i lo = AdjustedYuvToHue16(AdjustY16(UnpackU8<0>(y)), AdjustUV16(UnpackU8<0>(u)), AdjustUV16(UnpackU8<0>(v)), KF_255_DIV_6);
            __m512i hi = AdjustedYuvToHue16(AdjustY16(UnpackU8<1>(y)), AdjustUV16(UnpackU8<1>(u)), AdjustUV16(UnpackU8<1>(v)), KF_255_DIV_6);
            Store<align, mask>(hue, _mm512_packus_epi16(lo, hi), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv420pToHue(const uint8_t * y0, const uint8_t * y1, const uint8_t * u, const uint8_t * v,
            const __m512 & KF_255_DIV_6, uint8_t * hue0, uint8_t * hue1, const __mmask64 * tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToHue<align, mask>(Load<align, mask>(y0 + 0, tails[1]), u0, v0, KF_255_DIV_6, hue0 + 0, tails[1]);
            YuvToHue<align, mask>(Load<align, mask>(y0 + A, tails[2]), u1, v1, KF_255_DIV_6, hue0 + A, tails[2]);
            YuvToHue<align, mask>(Load<align, mask>(y1 + 0, tails[1]), u0, v0, KF_255_DIV_6, hue1 + 0, tails[1]);
            YuvToHue<align, mask>(Load<align, mask>(y1 + A, tails[2]), u1, v1, KF_255_DIV_6, hue1 + A, tails[2]);
        }

        template <bool align> void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride));
            }

            const __m512 KF_255_DIV_6 = _mm512_set1_ps(Base::KF_255_DIV_6);

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[3];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv420pToHue<align, false>(y + col * 2, y + yStride + col * 2, u + col, v + col, KF_255_DIV_6, hue + col * 2, hue + hueStride + col * 2, tailMasks);
                if (col < width)
                    Yuv420pToHue<align, true>(y + col * 2, y + yStride + col * 2, u + col, v + col, KF_255_DIV_6, hue + col * 2, hue + hueStride + col * 2, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                hue += 2 * hueStride;
            }
        }

        void Yuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hue, size_t hueStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride) && Aligned(hue) && Aligned(hueStride))
                Yuv420pToHue<true>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
            else
                Yuv420pToHue<false>(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv444pToHue(const uint8_t * y, const uint8_t * u, const uint8_t * v, const __m512 & KF_255_DIV_6, uint8_t * hue, __mmask64 tail = -1)
        {
            YuvToHue<align, mask>(Load<align, mask>(y, tail), Load<align, mask>(u, tail), Load<align, mask>(v, tail), KF_255_DIV_6, hue, tail);
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

            const __m512 KF_255_DIV_6 = _mm512_set1_ps(Base::KF_255_DIV_6);

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMask = TailMask64(tail);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToHue<align, false>(y + col, u + col, v + col, KF_255_DIV_6, hue + col);
                if (col < width)
                    Yuv444pToHue<align, true>(y + col, u + col, v + col, KF_255_DIV_6, hue + col, tailMask);
                y += yStride;
                u += uStride;
                v += vStride;
                hue += hueStride;
            }
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
#endif// SIMD_AVX512BW_ENABLE
}
