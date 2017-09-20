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
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
        SIMD_INLINE __m512i TextureBoostedSaturatedGradient16(const __m512i & difference, const __m512i & saturation, const __m512i & boost)
        {
            return _mm512_mullo_epi16(_mm512_max_epi16(K_ZERO, _mm512_add_epi16(saturation, _mm512_min_epi16(difference, saturation))), boost);
        }

        SIMD_INLINE __m512i TextureBoostedSaturatedGradient8(const __m512i & a, const __m512i & b, const __m512i & saturation, const __m512i & boost)
        {
            __m512i lo = TextureBoostedSaturatedGradient16(SubUnpackedU8<0>(b, a), saturation, boost);
            __m512i hi = TextureBoostedSaturatedGradient16(SubUnpackedU8<1>(b, a), saturation, boost);
            return _mm512_packus_epi16(lo, hi);
        }

        template<bool align, bool mask> SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, uint8_t * dx, uint8_t * dy, 
            size_t stride, const __m512i & saturation, const __m512i & boost, __mmask64 tail = -1)
        {
            const __m512i s10 = Load<false, mask>(src - 1, tail);
            const __m512i s12 = Load<false, mask>(src + 1, tail);
            const __m512i s01 = Load<align, mask>(src - stride, tail);
            const __m512i s21 = Load<align, mask>(src + stride, tail);
            Store<align, mask>(dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost), tail);
            Store<align, mask>(dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost), tail);
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(int(2)*saturation*boost <= 0xFF);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));

            size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			__m512i _saturation = _mm512_set1_epi16(saturation);
            __m512i _boost = _mm512_set1_epi16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
				size_t col = 0;
                for (; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align, false>(src + col, dx + col, dy + col, srcStride, _saturation, _boost);
                if(col < width)
                    TextureBoostedSaturatedGradient<false, true>(src + col, dx + col, dy + col, srcStride, _saturation, _boost, tailMask);

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

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }

		template<bool align, bool mask> SIMD_INLINE void TextureBoostedUv(const uint8_t * src, uint8_t * dst, 
			const __m512i & min8, const __m512i & max8, const __m512i & boost16, __mmask64 tail = -1)
		{
			const __m512i _src = Load<align, mask>(src, tail);
			const __m512i saturated = _mm512_sub_epi8(_mm512_max_epu8(min8, _mm512_min_epu8(max8, _src)), min8);
			const __m512i lo = _mm512_mullo_epi16(_mm512_unpacklo_epi8(saturated, K_ZERO), boost16);
			const __m512i hi = _mm512_mullo_epi16(_mm512_unpackhi_epi8(saturated, K_ZERO), boost16);
			Store<align, mask>(dst, _mm512_packus_epi16(lo, hi), tail);
		}

		template<bool align> void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			uint8_t boost, uint8_t * dst, size_t dstStride)
		{
			assert(boost < 0x80);
			if (align)
				assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			int min = 128 - (128 / boost);
			int max = 255 - min;
			__m512i min8 = _mm512_set1_epi8(min);
			__m512i max8 = _mm512_set1_epi8(max);
			__m512i boost16 = _mm512_set1_epi16(boost);
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < alignedWidth; col += A)
					TextureBoostedUv<align, false>(src + col, dst + col, min8, max8, boost16);
				if (col < width)
					TextureBoostedUv<false, true>(src + col, dst + col, min8, max8, boost16, tailMask);
				src += srcStride;
				dst += dstStride;
			}
		}

		void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
			uint8_t boost, uint8_t * dst, size_t dstStride)
		{
			if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
				TextureBoostedUv<true>(src, srcStride, width, height, boost, dst, dstStride);
			else
				TextureBoostedUv<false>(src, srcStride, width, height, boost, dst, dstStride);
		}
    }
#endif// SIMD_AVX512BW_ENABLE
}
