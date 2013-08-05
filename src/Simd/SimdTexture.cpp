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
	namespace Base
	{
        SIMD_INLINE int TextureBoostedSaturatedGradient(const uchar * src, ptrdiff_t step, int saturation, int boost)
        {
            return (saturation + RestrictRange((int)src[step] - (int)src[-step], -saturation, saturation))*boost;
        }

        void TextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
		{
            assert(int(2)*saturation*boost <= 0xFF);

			memset(dx, 0, width);
            memset(dy, 0, width);
			src += srcStride;
			dx += dxStride;
            dy += dyStride;
			for (size_t row = 2; row < height; ++row)
			{
				dx[0] = 0;
                dy[0] = 0;
				for (size_t col = 1; col < width - 1; ++col)
				{
					dy[col] = TextureBoostedSaturatedGradient(src + col, srcStride, saturation, boost);
					dx[col] = TextureBoostedSaturatedGradient(src + col, 1, saturation, boost);
				}
				dx[width - 1] = 0;
                dy[width - 1] = 0;
				src += srcStride;
				dx += dxStride;
                dy += dyStride;
			}
			memset(dx, 0, width);
            memset(dy, 0, width);
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
        SIMD_INLINE __m128i TextureBoostedSaturatedGradient16(__m128i a, __m128i b, __m128i saturation, const __m128i & boost)
        {
            return _mm_mullo_epi16(_mm_max_epi16(K_ZERO, _mm_add_epi16(saturation, _mm_min_epi16(_mm_sub_epi16(b, a), saturation))), boost);
        }

        SIMD_INLINE __m128i TextureBoostedSaturatedGradient8(__m128i a, __m128i b, __m128i saturation, const __m128i & boost) 
        {
            __m128i lo = TextureBoostedSaturatedGradient16(_mm_unpacklo_epi8(a, K_ZERO), _mm_unpacklo_epi8(b, K_ZERO), saturation, boost);
            __m128i hi = TextureBoostedSaturatedGradient16(_mm_unpackhi_epi8(a, K_ZERO), _mm_unpackhi_epi8(b, K_ZERO), saturation, boost);
            return _mm_packus_epi16(lo, hi);
        }

		template<bool align> SIMD_INLINE void TextureBoostedSaturatedGradient(const uchar * src, uchar * dx, uchar * dy, 
            size_t stride, __m128i saturation, __m128i boost)
		{
			const __m128i s10 = Load<false>((__m128i*)(src - 1));
			const __m128i s12 = Load<false>((__m128i*)(src + 1));
			const __m128i s01 = Load<align>((__m128i*)(src - stride));
			const __m128i s21 = Load<align>((__m128i*)(src + stride));
			Store<align>((__m128i*)dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost));
			Store<align>((__m128i*)dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost));
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
            __m128i _saturation = _mm_set1_epi16(saturation);
            __m128i _boost = _mm_set1_epi16(boost);

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
#endif// SIMD_SSE2_ENABLE

    void TextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
        uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
		else
#endif//SIMD_SSE2_ENABLE
			Base::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
	}

    void TextureBoostedSaturatedGradient(const View & src, uchar saturation, uchar boost, View &  dx, View & dy)
	{
		assert(src.width == dx.width && src.height == dx.height && src.format == dx.format);
        assert(src.width == dy.width && src.height == dy.height && src.format == dy.format);
		assert(src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

		TextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
	}
}