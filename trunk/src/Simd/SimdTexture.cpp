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
#include "Simd/SimdExtract.h"
#include "Simd/SimdCopy.h"
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

        void TextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar boost, uchar * dst, size_t dstStride)
        {
            assert(boost < 128);

            int min = 128 - (128/boost);
            int max = 255 - min;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    dst[col] = (RestrictRange(src[col], min, max) - min)*boost;
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void TextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
        {
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                    rowSum += src[col] - Average(lo[col], hi[col]);
                *sum += rowSum;

                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void TexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
            int shift, uchar * dst, size_t dstStride)
        {
            assert(shift > -0xFF && shift < 0xFF);

            if(shift == 0)
            {
                if(src != dst)
                    Copy(src, srcStride, width, height, 1, dst, dstStride);
                return;
            }
            else if(shift > 0)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                        dst[col] = Min(src[col] + shift, 0xFF);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else if(shift < 0)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                        dst[col] = Max(src[col] + shift, 0);
                    src += srcStride;
                    dst += dstStride;
                }
            }
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
            assert(width >= A && int(2)*saturation*boost <= 0xFF);
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

        template<bool align> SIMD_INLINE void TextureBoostedUv(const uchar * src, uchar * dst, __m128i min8, __m128i max8, __m128i boost16)
        {
            const __m128i _src = Load<align>((__m128i*)src);
            const __m128i saturated = _mm_sub_epi8(_mm_max_epu8(min8, _mm_min_epu8(max8, _src)), min8);
            const __m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(saturated, K_ZERO), boost16);
            const __m128i hi = _mm_mullo_epi16(_mm_unpackhi_epi8(saturated, K_ZERO), boost16);
            Store<align>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template<bool align> void TextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar boost, uchar * dst, size_t dstStride)
        {
            assert(width >= A && boost < 0x80);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            int min = 128 - (128/boost);
            int max = 255 - min;

            __m128i min8 = _mm_set1_epi8(min);
            __m128i max8 = _mm_set1_epi8(max);
            __m128i boost16 = _mm_set1_epi16(boost);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedUv<align>(src + col, dst + col, min8, max8, boost16);
                if(width != alignedWidth)
                    TextureBoostedUv<false>(src + width - A, dst + width - A, min8, max8, boost16);

                src += srcStride;
                dst += dstStride;
            }
        }

        void TextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar boost, uchar * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TextureBoostedUv<true>(src, srcStride, width, height, boost, dst, dstStride);
            else
                TextureBoostedUv<false>(src, srcStride, width, height, boost, dst, dstStride);
        }

        template <bool align> SIMD_INLINE void TextureGetDifferenceSum(const uchar * src, const uchar * lo, const uchar * hi, 
            __m128i & positive, __m128i & negative, const __m128i & mask)
        {
            const __m128i _src = Load<align>((__m128i*)src);
            const __m128i _lo = Load<align>((__m128i*)lo);
            const __m128i _hi = Load<align>((__m128i*)hi);
            const __m128i average = _mm_and_si128(mask, _mm_avg_epu8(_lo, _hi));
            const __m128i current = _mm_and_si128(mask, _src);
            positive = _mm_add_epi64(positive, _mm_sad_epu8(_mm_subs_epu8(current, average), K_ZERO));
            negative = _mm_add_epi64(negative, _mm_sad_epu8(_mm_subs_epu8(average, current), K_ZERO));
        }

        template <bool align> void TextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
        {
            assert(width >= A && sum != NULL);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            __m128i positive = _mm_setzero_si128();
            __m128i negative = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureGetDifferenceSum<align>(src + col, lo + col, hi + col, positive, negative, K_INV_ZERO);
                if(width != alignedWidth)
                    TextureGetDifferenceSum<false>(src + width - A, lo + width - A, hi + width - A, positive, negative, tailMask);
                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
            *sum = ExtractInt64Sum(positive) - ExtractInt64Sum(negative);
        }

        void TextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                TextureGetDifferenceSum<true>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
            else
                TextureGetDifferenceSum<false>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        }

        template <bool align> void TexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
            int shift, uchar * dst, size_t dstStride)
        {
            assert(width >= A && shift > -0xFF && shift < 0xFF && shift != 0);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = src == dst ? ShiftLeft(K_INV_ZERO, A - width + alignedWidth) : K_INV_ZERO;
            if(shift > 0)
            {
                __m128i _shift = _mm_set1_epi8((char)shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        const __m128i _src = Load<align>((__m128i*) (src + col));
                        Store<align>((__m128i*) (dst + col),  _mm_adds_epu8(_src, _shift));
                    }
                    if(width != alignedWidth)
                    {
                        const __m128i _src = Load<false>((__m128i*) (src + width - A));
                        Store<false>((__m128i*) (dst + width - A),  _mm_adds_epu8(_src, _mm_and_si128(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
            if(shift < 0)
            {
                __m128i _shift = _mm_set1_epi8((char)-shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        const __m128i _src = Load<align>((__m128i*) (src + col));
                        Store<align>((__m128i*) (dst + col),  _mm_subs_epu8(_src, _shift));
                    }
                    if(width != alignedWidth)
                    {
                        const __m128i _src = Load<false>((__m128i*) (src + width - A));
                        Store<false>((__m128i*) (dst + width - A),  _mm_subs_epu8(_src, _mm_and_si128(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
        }

        void TexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
            int shift, uchar * dst, size_t dstStride)
        {
            if(shift == 0)
            {
                if(src != dst)
                    Copy(src, srcStride, width, height, 1, dst, dstStride);
                return;
            }
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TexturePerformCompensation<true>(src, srcStride, width, height, shift, dst, dstStride);
            else
                TexturePerformCompensation<false>(src, srcStride, width, height, shift, dst, dstStride);
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

    void TextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
        uchar boost, uchar * dst, size_t dstStride)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
        else
#endif//SIMD_SSE2_ENABLE
            Base::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    }

    void TextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
        const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        else
#endif//SIMD_SSE2_ENABLE
            Base::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    }

    void TexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
        int shift, uchar * dst, size_t dstStride)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
        else
#endif//SIMD_SSE2_ENABLE
            Base::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    }

    void TextureBoostedSaturatedGradient(const View & src, uchar saturation, uchar boost, View &  dx, View & dy)
	{
		assert(src.width == dx.width && src.height == dx.height && src.format == dx.format);
        assert(src.width == dy.width && src.height == dy.height && src.format == dy.format);
		assert(src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

		TextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
	}

    void TextureBoostedUv(const View & src, uchar boost, View & dst)
    {
        assert(src.width == dst.width && src.height == dst.height && src.format == dst.format && src.format == View::Gray8);

        TextureBoostedUv(src.data, src.stride, src.width, src.height, boost, dst.data, dst.stride);
    }

    void TextureGetDifferenceSum(const View & src, const View & lo, const View & hi, int64_t * sum)
    {
        assert(src.width == lo.width && src.height == lo.height && src.format == lo.format);
        assert(src.width == hi.width && src.height == hi.height && src.format == hi.format);
        assert(src.format == View::Gray8 && sum != NULL);

        TextureGetDifferenceSum(src.data, src.stride, src.width, src.height, lo.data, lo.stride, hi.data, hi.stride, sum);
    }

    void TexturePerformCompensation(const View & src, int shift, View & dst)
    {
        assert(src.width == dst.width && src.height == dst.height && src.format == dst.format && src.format == View::Gray8);
        assert(shift > -0xFF && shift < 0xFF);

        TexturePerformCompensation(src.data, src.stride, src.width, src.height, shift, dst.data, dst.stride);
    }
}