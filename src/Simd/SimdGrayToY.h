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
#ifndef __SimdGrayToY_h__
#define __SimdGrayToY_h__

#include "Simd/SimdInit.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
    namespace Base
    {
        const int G2Y_LO = 16;
        const int G2Y_HI = 235;
        const int Y2G_LO = 0;
        const int Y2G_HI = 255;

        const int G2Y_SHIFT = 8;
        const int G2Y_RANGE = 1 << G2Y_SHIFT;
        const int G2Y_ROUND = 1 << (G2Y_SHIFT - 1);
        const int G2Y_SCALE = int(G2Y_RANGE * (G2Y_HI - G2Y_LO) / (Y2G_HI - Y2G_LO) + 0.5f);

        const int Y2G_SHIFT = 8;
        const int Y2G_RANGE = 1 << Y2G_SHIFT;
        const int Y2G_ROUND = 1 << (Y2G_SHIFT - 1);
        const int Y2G_SCALE = int(Y2G_RANGE * (Y2G_HI - Y2G_LO) / (G2Y_HI - G2Y_LO) + 0.5f);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int GrayToY(int g)
        {
            int y = ((G2Y_SCALE * g + G2Y_ROUND) >> G2Y_SHIFT) + G2Y_LO;
            return RestrictRange(y, G2Y_LO, G2Y_HI);
        }

        SIMD_INLINE int YToGray(int y)
        {
            y = RestrictRange(y, G2Y_LO, G2Y_HI);
            int g = (Y2G_SCALE * (y - G2Y_LO) + Y2G_ROUND) >> Y2G_SHIFT;
            return RestrictRange(g, Y2G_LO, Y2G_HI);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i GrayToY(__m128i g)
        {
            static const __m128i G2Y_SCALE = SIMD_MM_SET1_EPI16(Base::G2Y_SCALE);
            static const __m128i G2Y_ROUND = SIMD_MM_SET1_EPI16(Base::G2Y_ROUND);
            static const __m128i G2Y_LO = SIMD_MM_SET1_EPI8(Base::G2Y_LO);
            static const __m128i G2Y_HI = SIMD_MM_SET1_EPI8(Base::G2Y_HI);
            __m128i g0 = UnpackU8<0>(g);
            __m128i g1 = UnpackU8<1>(g);
            __m128i y0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(g0, G2Y_SCALE), G2Y_ROUND), Base::G2Y_SHIFT);
            __m128i y1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(g1, G2Y_SCALE), G2Y_ROUND), Base::G2Y_SHIFT);
            __m128i y = _mm_packus_epi16(y0, y1);
            return _mm_min_epu8(_mm_adds_epu8(y, G2Y_LO), G2Y_HI);
        }

        SIMD_INLINE __m128i YToGray(__m128i y)
        {
            static const __m128i Y2G_SCALE = SIMD_MM_SET1_EPI16(Base::Y2G_SCALE);
            static const __m128i Y2G_ROUND = SIMD_MM_SET1_EPI16(Base::Y2G_ROUND);
            static const __m128i G2Y_LO = SIMD_MM_SET1_EPI8(Base::G2Y_LO);
            static const __m128i G2Y_HI = SIMD_MM_SET1_EPI8(Base::G2Y_HI);
            y = _mm_subs_epu8(_mm_min_epu8(y, G2Y_HI), G2Y_LO);
            __m128i y0 = UnpackU8<0>(y);
            __m128i y1 = UnpackU8<1>(y);
            __m128i g0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(y0, Y2G_SCALE), Y2G_ROUND), Base::Y2G_SHIFT);
            __m128i g1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(y1, Y2G_SCALE), Y2G_ROUND), Base::Y2G_SHIFT);
            return _mm_packus_epi16(g0, g1);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i GrayToY(__m256i g)
        {
            static const __m256i G2Y_SCALE = SIMD_MM256_SET1_EPI16(Base::G2Y_SCALE);
            static const __m256i G2Y_ROUND = SIMD_MM256_SET1_EPI16(Base::G2Y_ROUND);
            static const __m256i G2Y_LO = SIMD_MM256_SET1_EPI8(Base::G2Y_LO);
            static const __m256i G2Y_HI = SIMD_MM256_SET1_EPI8(Base::G2Y_HI);
            __m256i g0 = UnpackU8<0>(g);
            __m256i g1 = UnpackU8<1>(g);
            __m256i y0 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(g0, G2Y_SCALE), G2Y_ROUND), Base::G2Y_SHIFT);
            __m256i y1 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(g1, G2Y_SCALE), G2Y_ROUND), Base::G2Y_SHIFT);
            __m256i y = _mm256_packus_epi16(y0, y1);
            return _mm256_min_epu8(_mm256_adds_epu8(y, G2Y_LO), G2Y_HI);
        }

        SIMD_INLINE __m256i YToGray(__m256i y)
        {
            static const __m256i Y2G_SCALE = SIMD_MM256_SET1_EPI16(Base::Y2G_SCALE);
            static const __m256i Y2G_ROUND = SIMD_MM256_SET1_EPI16(Base::Y2G_ROUND);
            static const __m256i G2Y_LO = SIMD_MM256_SET1_EPI8(Base::G2Y_LO);
            static const __m256i G2Y_HI = SIMD_MM256_SET1_EPI8(Base::G2Y_HI);
            y = _mm256_subs_epu8(_mm256_min_epu8(y, G2Y_HI), G2Y_LO);
            __m256i y0 = UnpackU8<0>(y);
            __m256i y1 = UnpackU8<1>(y);
            __m256i g0 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(y0, Y2G_SCALE), Y2G_ROUND), Base::Y2G_SHIFT);
            __m256i g1 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(y1, Y2G_SCALE), Y2G_ROUND), Base::Y2G_SHIFT);
            return _mm256_packus_epi16(g0, g1);
        }
    }
#endif
}

#endif//__SimdGrayToY_h__
