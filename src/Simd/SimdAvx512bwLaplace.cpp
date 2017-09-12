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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
		const __m512i K64_PERMUTE_0 = SIMD_MM512_SETR_EPI64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
		const __m512i K64_PERMUTE_1 = SIMD_MM512_SETR_EPI64(0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF);

		template<int part> SIMD_INLINE __m512i Laplace(__m512i a[3][3])
        {
            return _mm512_sub_epi16(_mm512_mullo_epi16(K16_0008, UnpackU8<part>(a[1][1])),
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<part>(a[0][0], a[0][1]), K8_01),
                _mm512_maddubs_epi16(UnpackU8<part>(a[0][2], a[1][0]), K8_01)),
                _mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<part>(a[1][2], a[2][0]), K8_01),
                _mm512_maddubs_epi16(UnpackU8<part>(a[2][1], a[2][2]), K8_01))));
        }

        template<bool align, bool abs> SIMD_INLINE void Laplace(__m512i a[3][3], int16_t * dst)
        {
            __m512i lo = ConditionalAbs<abs>(Laplace<0>(a));
            __m512i hi = ConditionalAbs<abs>(Laplace<1>(a));
            Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
            Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
        }

        template <bool align, bool abs> void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m512i a[3][3];

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                Laplace<align, abs>(a, dst + 0);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    Laplace<align, abs>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                Laplace<false, abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Laplace<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                Laplace<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
