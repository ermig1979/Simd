/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K16_BLUE_RED = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);        
        const __m256i K16_GREEN_ROUND = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m256i BgraToGray32(__m256i bgra)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(bgra, 1), K16_00FF);
            const __m256i b0r0 = _mm256_and_si256(bgra, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_ROUND), _mm256_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm256_srli_epi32(weightedSum, Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgraToGray(__m256i bgra[4])
        {
            const __m256i lo = PackI32ToI16(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m256i hi = PackI32ToI16(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return PackU16ToU8(lo, hi);
        }

        SIMD_INLINE __m256i PermuteAndShiffle(__m256i bgr, __m256i permute, __m256i shuffle)
        {
            return _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(bgr, permute), shuffle);
        }

        template <bool align> SIMD_INLINE __m256i BgrToGray(const uint8_t * bgr, __m256i permuteBody, __m256i permuteTail, __m256i shuffle)
        {
            __m256i bgra[4];
            bgra[0] = _mm256_or_si256(K32_01000000, PermuteAndShiffle(Load<align>((__m256i*)(bgr + 0)), permuteBody, shuffle));
            bgra[1] = _mm256_or_si256(K32_01000000, PermuteAndShiffle(Load<false>((__m256i*)(bgr + 24)), permuteBody, shuffle));
            bgra[2] = _mm256_or_si256(K32_01000000, PermuteAndShiffle(Load<false>((__m256i*)(bgr + 48)), permuteBody, shuffle));
            bgra[3] = _mm256_or_si256(K32_01000000, PermuteAndShiffle(Load<align>((__m256i*)(bgr + 64)), permuteTail, shuffle));
            return BgraToGray(bgra);
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);

            __m256i _permuteBody = _mm256_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0);
            __m256i _permuteTail = _mm256_setr_epi32(2, 3, 4, 0, 5, 6, 7, 0);

            __m256i _shuffle = _mm256_setr_epi8(
                0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
                0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                    Store<align>((__m256i*)(gray + col), BgrToGray<align>(bgr + 3*col, _permuteBody, _permuteTail, _shuffle));
                if(width != alignedWidth)
                    Store<false>((__m256i*)(gray + width - A), BgrToGray<false>(bgr + 3*(width - A), _permuteBody, _permuteTail, _shuffle));
                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if(Aligned(gray) && Aligned(grayStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToGray<true>(bgr, width, height, bgrStride, gray, grayStride);
            else
                BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }
    }
#endif//SIMD_Avx2_ENABLE
}