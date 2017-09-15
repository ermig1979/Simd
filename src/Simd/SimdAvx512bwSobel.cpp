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

		template<bool abs> SIMD_INLINE void SobelDx(__m512i a[3][3], __m512i & lo, __m512i & hi)
		{
			lo = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<0>(a[0][2], a[0][0]), SubUnpackedU8<0>(a[1][2], a[1][0]), SubUnpackedU8<0>(a[2][2], a[2][0])));
			hi = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<1>(a[0][2], a[0][0]), SubUnpackedU8<1>(a[1][2], a[1][0]), SubUnpackedU8<1>(a[2][2], a[2][0])));
		}

        template<bool align, bool abs> SIMD_INLINE void SobelDx(__m512i a[3][3], int16_t * dst)
        {
			__m512i lo, hi;
			SobelDx<abs>(a, lo, hi);
			Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
			Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
		}

        template <bool align, bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if(align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

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

                LoadNoseDx(src0 + 0, a[0]);
                LoadNoseDx(src1 + 0, a[1]);
                LoadNoseDx(src2 + 0, a[2]);
                SobelDx<align, abs>(a, dst + 0);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBodyDx(src0 + col, a[0]);
                    LoadBodyDx(src1 + col, a[1]);
                    LoadBodyDx(src2 + col, a[2]);
                    SobelDx<align, abs>(a, dst + col);
                }
                LoadTailDx(src0 + width - A, a[0]);
                LoadTailDx(src1 + width - A, a[1]);
                LoadTailDx(src2 + width - A, a[2]);
                SobelDx<false, abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(dst) && Aligned(dstStride))
                SobelDx<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                SobelDx<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }

		void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
		{
			assert(dstStride % sizeof(int16_t) == 0);

			if (Aligned(dst) && Aligned(dstStride))
				SobelDx<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
			else
				SobelDx<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
		}

		SIMD_INLINE void SobelDxAbsSum(__m512i a[3][3], __m512i * sums)
		{
			__m512i lo, hi;
			SobelDx<true>(a, lo, hi);
			sums[0] = _mm512_add_epi32(sums[0], _mm512_madd_epi16(lo, K16_0001));
			sums[1] = _mm512_add_epi32(sums[1], _mm512_madd_epi16(hi, K16_0001));
		}

		SIMD_INLINE void SetMask3(__m512i a[3], __m512i mask)
		{
			a[0] = _mm512_and_si512(a[0], mask);
			a[1] = _mm512_and_si512(a[1], mask);
			a[2] = _mm512_and_si512(a[2], mask);
		}

		SIMD_INLINE void SetMask3x3(__m512i a[3][3], __m512i mask)
		{
			SetMask3(a[0], mask);
			SetMask3(a[1], mask);
			SetMask3(a[2], mask);
		}

		template <bool align> void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width > A && width < 256 * 256 * F);
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t bodyWidth = Simd::AlignHi(width, A) - A;
			const uint8_t *src0, *src1, *src2;

			__m512i a[3][3];
			__m512i tailMask = _mm512_mask_set1_epi8(K_INV_ZERO, TailMask64(A - width + bodyWidth), 0);

			size_t blockSize = (256 * 256 * F) / width;
			size_t blockCount = height / blockSize + 1;
			__m512i _sum = _mm512_setzero_si512();
			for (size_t block = 0; block < blockCount; ++block)
			{
				__m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
				for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
				{
					src0 = src + stride*(row - 1);
					src1 = src0 + stride;
					src2 = src1 + stride;
					if (row == 0)
						src0 = src1;
					if (row == height - 1)
						src2 = src1;

					LoadNoseDx(src0 + 0, a[0]);
					LoadNoseDx(src1 + 0, a[1]);
					LoadNoseDx(src2 + 0, a[2]);
					SobelDxAbsSum(a, sums);
					for (size_t col = A; col < bodyWidth; col += A)
					{
						LoadBodyDx(src0 + col, a[0]);
						LoadBodyDx(src1 + col, a[1]);
						LoadBodyDx(src2 + col, a[2]);
						SobelDxAbsSum(a, sums);
					}
					LoadTailDx(src0 + width - A, a[0]);
					LoadTailDx(src1 + width - A, a[1]);
					LoadTailDx(src2 + width - A, a[2]);
					SetMask3x3(a, tailMask);
					SobelDxAbsSum(a, sums);
				}
				sums[0] = _mm512_add_epi32(sums[0], sums[1]);
				_sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
			}

			*sum = ExtractSum<uint64_t>(_sum);
		}

		void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(src) && Aligned(stride))
				SobelDxAbsSum<true>(src, stride, width, height, sum);
			else
				SobelDxAbsSum<false>(src, stride, width, height, sum);
		}
    }
#endif// SIMD_AVX512BW_ENABLE
}
