/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdTexture.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
        SIMD_INLINE __m256i TextureBoostedSaturatedGradient16(__m256i a, __m256i b, __m256i saturation, const __m256i & boost)
        {
            return _mm256_mullo_epi16(_mm256_max_epi16(K_ZERO, _mm256_add_epi16(saturation, _mm256_min_epi16(_mm256_sub_epi16(b, a), saturation))), boost);
        }

        SIMD_INLINE __m256i TextureBoostedSaturatedGradient8(__m256i a, __m256i b, __m256i saturation, const __m256i & boost) 
        {
            __m256i lo = TextureBoostedSaturatedGradient16(_mm256_unpacklo_epi8(a, K_ZERO), _mm256_unpacklo_epi8(b, K_ZERO), saturation, boost);
            __m256i hi = TextureBoostedSaturatedGradient16(_mm256_unpackhi_epi8(a, K_ZERO), _mm256_unpackhi_epi8(b, K_ZERO), saturation, boost);
            return _mm256_packus_epi16(lo, hi);
        }

        template<bool align> SIMD_INLINE void TextureBoostedSaturatedGradient(const uchar * src, uchar * dx, uchar * dy, 
            size_t stride, __m256i saturation, __m256i boost)
        {
            const __m256i s10 = Load<false>((__m256i*)(src - 1));
            const __m256i s12 = Load<false>((__m256i*)(src + 1));
            const __m256i s01 = Load<align>((__m256i*)(src - stride));
            const __m256i s21 = Load<align>((__m256i*)(src + stride));
            Store<align>((__m256i*)dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost));
            Store<align>((__m256i*)dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost));
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m256i _saturation = _mm256_set1_epi16(saturation);
            __m256i _boost = _mm256_set1_epi16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align>(src + col, dx + col, dy + col, srcStride, _saturation, _boost);
                if(width != alignedWidth)
                    TextureBoostedSaturatedGradient<false>(src + width - A, dx + width - A, dy + width - A, srcStride, _saturation, _boost);

                dx[0] = 0;
                dy[0] = 0;
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}