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
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdEdgeBackground.h"

namespace Simd
{
	namespace Base
	{
        void EdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * background, size_t backgroundStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    if (value[col] > background[col])
                        background[col]++;
                }
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * background, size_t backgroundStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    background[col] = MaxU8(value[col], background[col]);
                }
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
            const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    if(value[col] > backgroundValue[col] && backgroundCount[col] < 0xFF)
                        backgroundCount[col]++;
                }
                value += valueStride;
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        SIMD_INLINE void AdjustEdge(const uchar & count, uchar & value, uchar threshold)
        {
            if(count < threshold)
            {
                if(value > 0)
                    value--;
            }
            else if(count > threshold)
            {
                if(value < 0xFF)
                    value++;
            }
        }

        void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
            uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    AdjustEdge(backgroundCount[col], backgroundValue[col], threshold);
                    backgroundCount[col] = 0;
                }
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
            }
        }

        void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
            uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    if(mask[col])
                        AdjustEdge(backgroundCount[col], backgroundValue[col], threshold);
                    backgroundCount[col] = 0;
                }
                backgroundValue += backgroundValueStride;
                backgroundCount += backgroundCountStride;
                mask += maskStride;
            }
        }

        void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * background, size_t backgroundStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                    background[col] = value[col];
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
            uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                {
                    if(mask[col])
                        background[col] = value[col];
                }
                value += valueStride;
                background += backgroundStride;
                mask += maskStride;
            }
        }
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const uchar * value, uchar * background, __m128i tailMask)
		{
			const __m128i _value = Load<align>((__m128i*)value);
			const __m128i _background = Load<align>((__m128i*)background);
			const __m128i inc = _mm_and_si128(tailMask, GreaterThenU8(_value, _background));
			Store<align>((__m128i*)background, _mm_adds_epu8(_background, inc));
		}

		template <bool align> void EdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(background) && Aligned(backgroundStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundGrowRangeSlow<align>(value + col, background + col, K8_01);
				if(alignedWidth != width)
					EdgeBackgroundGrowRangeSlow<false>(value + width - A, background + width - A, tailMask);
				value += valueStride;
				background += backgroundStride;
			}
		}

		void EdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
				EdgeBackgroundGrowRangeSlow<true>(value, valueStride, width, height, background, backgroundStride);
			else
				EdgeBackgroundGrowRangeSlow<false>(value, valueStride, width, height, background, backgroundStride);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const uchar * value, uchar * background)
		{
			const __m128i _value = Load<align>((__m128i*)value);
			const __m128i _background = Load<align>((__m128i*)background);
			Store<align>((__m128i*)background, _mm_max_epu8(_background, _value));
		}

		template <bool align> void EdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(background) && Aligned(backgroundStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundGrowRangeFast<align>(value + col, background + col);
				if(alignedWidth != width)
					EdgeBackgroundGrowRangeFast<false>(value + width - A, background + width - A);
				value += valueStride;
				background += backgroundStride;
			}
		}

		void EdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
				EdgeBackgroundGrowRangeFast<true>(value, valueStride, width, height, background, backgroundStride);
			else
				EdgeBackgroundGrowRangeFast<false>(value, valueStride, width, height, background, backgroundStride);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundIncrementCount(const uchar * value, 
			const uchar * backgroundValue, uchar * backgroundCount, size_t offset, __m128i tailMask)
		{
			const __m128i _value = Load<align>((__m128i*)(value + offset));
			const __m128i _backgroundValue = Load<align>((__m128i*)(backgroundValue + offset));
			const __m128i _backgroundCount = Load<align>((__m128i*)(backgroundCount + offset));

			const __m128i inc = _mm_and_si128(tailMask, GreaterThenU8(_value, _backgroundValue));

			Store<align>((__m128i*)(backgroundCount + offset), _mm_adds_epu8(_backgroundCount, inc));
		}

		template <bool align> void EdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundIncrementCount<align>(value, backgroundValue, backgroundCount, col, K8_01);
				if(alignedWidth != width)
					EdgeBackgroundIncrementCount<false>(value, backgroundValue, backgroundCount, width - A, tailMask);
				value += valueStride;
				backgroundValue += backgroundValueStride;
				backgroundCount += backgroundCountStride;
			}
		}

		void EdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
			const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
		{
			if(Aligned(value) && Aligned(valueStride) && 
				Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride))
				EdgeBackgroundIncrementCount<true>(value, valueStride, width, height,
				backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
			else
				EdgeBackgroundIncrementCount<false>(value, valueStride, width, height,
				backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
		}

		SIMD_INLINE __m128i AdjustEdge(const __m128i & count, const __m128i & value, const __m128i & mask, const __m128i & threshold)
		{
            const __m128i inc = _mm_and_si128(mask, GreaterThenU8(count, threshold));
            const __m128i dec = _mm_and_si128(mask, LesserThenU8(count, threshold));
            return _mm_subs_epu8(_mm_adds_epu8(value, inc), dec);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundAdjustRange(uchar * backgroundCount, uchar * backgroundValue, 
			size_t offset, const __m128i & threshold, const __m128i & mask)
		{
			const __m128i _backgroundCount = Load<align>((__m128i*)(backgroundCount + offset));
			const __m128i _backgroundValue = Load<align>((__m128i*)(backgroundValue + offset));

			Store<align>((__m128i*)(backgroundValue + offset), AdjustEdge(_backgroundCount, _backgroundValue, mask, threshold));
			Store<align>((__m128i*)(backgroundCount + offset), K_ZERO);
		}

		template <bool align> void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
			uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride));
			}

			const __m128i _threshold = _mm_set1_epi8((char)threshold);
			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundAdjustRange<align>(backgroundCount, backgroundValue, col, _threshold, K8_01);
				if(alignedWidth != width)
					EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundValue, width - A, _threshold, tailMask);
				backgroundValue += backgroundValueStride;
				backgroundCount += backgroundCountStride;
			}
		}

		void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
			uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
		{
			if(	Aligned(backgroundValue) && Aligned(backgroundValueStride) && 
				Aligned(backgroundCount) && Aligned(backgroundCountStride))
				EdgeBackgroundAdjustRange<true>(backgroundCount, backgroundCountStride, width, height, 
                backgroundValue, backgroundValueStride, threshold);
			else
                EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundCountStride, width, height, 
                backgroundValue, backgroundValueStride, threshold);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundAdjustRange(uchar * backgroundCount, uchar * backgroundValue, 
			const uchar * mask, size_t offset, const __m128i & threshold, const __m128i & tailMask)
		{
			const __m128i _mask = Load<align>((const __m128i*)(mask + offset));
			EdgeBackgroundAdjustRange<align>(backgroundCount, backgroundValue, offset, threshold, _mm_and_si128(_mask, tailMask));
		}

		template <bool align> void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
			uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(backgroundValue) && Aligned(backgroundValueStride));
				assert(Aligned(backgroundCount) && Aligned(backgroundCountStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			const __m128i _threshold = _mm_set1_epi8((char)threshold);
			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundAdjustRange<align>(backgroundCount, backgroundValue, mask, col, _threshold, K8_01);
				if(alignedWidth != width)
					EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundValue, mask, width - A, _threshold, tailMask);
				backgroundValue += backgroundValueStride;
				backgroundCount += backgroundCountStride;
				mask += maskStride;
			}		
		}

		void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
			uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
		{
			if(	Aligned(backgroundValue) && Aligned(backgroundValueStride) && Aligned(backgroundCount) && Aligned(backgroundCountStride) && 
                Aligned(mask) && Aligned(maskStride))
				EdgeBackgroundAdjustRange<true>(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, 
				threshold, mask, maskStride);
			else
                EdgeBackgroundAdjustRange<false>(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, 
                threshold, mask, maskStride);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundShiftRange(const uchar * value, uchar * background, size_t offset, __m128i mask)
		{
			const __m128i _value = Load<align>((__m128i*)(value + offset));
			const __m128i _background = Load<align>((__m128i*)(background + offset));
			Store<align>((__m128i*)(background + offset), _mm_or_si128(_mm_and_si128(mask, _value), _mm_andnot_si128(mask, _background)));
		}

		template <bool align> void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(background) && Aligned(backgroundStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundShiftRange<align>(value, background, col, K_INV_ZERO);
				if(alignedWidth != width)
					EdgeBackgroundShiftRange<false>(value, background, width - A, tailMask);
				value += valueStride;
				background += backgroundStride;
			}
		}

		void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
				EdgeBackgroundShiftRange<true>(value, valueStride, width, height, background, backgroundStride);
			else
				EdgeBackgroundShiftRange<false>(value, valueStride, width, height, background, backgroundStride);
		}

		template <bool align> SIMD_INLINE void EdgeBackgroundShiftRange(const uchar * value, uchar * background, const uchar * mask, 
			size_t offset, __m128i tailMask)
		{
			const __m128i _mask = Load<align>((const __m128i*)(mask + offset));
			EdgeBackgroundShiftRange<align>(value, background, offset, _mm_and_si128(_mask, tailMask));
		}

		template <bool align> void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(background) && Aligned(backgroundStride));
				assert(Aligned(mask) && Aligned(maskStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundShiftRange<align>(value, background, mask, col, K_INV_ZERO);
				if(alignedWidth != width)
					EdgeBackgroundShiftRange<false>(value, background, mask, width - A, tailMask);
				value += valueStride;
				background += backgroundStride;
				mask += maskStride;
			}
		}

		void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
			uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
		{
			if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride) && Aligned(mask) && Aligned(maskStride))
				EdgeBackgroundShiftRange<true>(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
			else
				EdgeBackgroundShiftRange<false>(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

    void EdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
	}

    void EdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
	}

    void EdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
        const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
            backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
			backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::EdgeBackgroundIncrementCount(value, valueStride, width, height,
			backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
	}

    void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
        uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
            Sse2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold);
		else
#endif// SIMD_SSE2_ENABLE
            Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold);
	}

    void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
        uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold, mask, maskStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
            Sse2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold, mask, maskStride);
		else
#endif// SIMD_SSE2_ENABLE
            Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
            backgroundValue, backgroundValueStride, threshold, mask, maskStride);
	}

    void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
	}

    void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
		else
#endif// SIMD_SSE2_ENABLE
			Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
	}

	void EdgeBackgroundGrowRangeSlow(const View & value, View & background)
	{
		assert(value.width == background.width && value.height == background.height);
		assert(value.format == View::Gray8 && background.format == View::Gray8);

		EdgeBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, background.data, background.stride);
	}

	void EdgeBackgroundGrowRangeFast(const View & value, View & background)
	{
        assert(value.width == background.width && value.height == background.height);
        assert(value.format == View::Gray8 && background.format == View::Gray8);

		EdgeBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, background.data, background.stride);
	}

	void EdgeBackgroundIncrementCount(const View & value, const View & backgroundValue, View & backgroundCount)
	{
		assert(value.width == backgroundValue.width && value.height == backgroundValue.height && 
			value.width == backgroundCount.width && value.height == backgroundCount.height);
		assert(value.format == View::Gray8 && backgroundValue.format == View::Gray8 && backgroundCount.format == View::Gray8);

		EdgeBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
			backgroundValue.data, backgroundValue.stride, backgroundCount.data, backgroundCount.stride);
	}

	void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold)
	{
		assert(backgroundValue.width == backgroundCount.width && backgroundValue.height == backgroundCount.height);
		assert(backgroundValue.format == View::Gray8 && backgroundCount.format == View::Gray8);

		EdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height, 
			backgroundValue.data, backgroundValue.stride, threshold);
	}

	void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold, const View & mask)
	{
		assert(backgroundValue.width == backgroundCount.width && backgroundValue.height == backgroundCount.height &&
			backgroundValue.width == mask.width && backgroundValue.height == mask.height);
		assert(backgroundValue.format == View::Gray8 && backgroundCount.format == View::Gray8 && mask.format == View::Gray8);

        EdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height, 
            backgroundValue.data, backgroundValue.stride, threshold, mask.data, mask.stride);
	}

	void EdgeBackgroundShiftRange(const View & value, View & background)
	{
		assert(value.width == background.width && value.height == background.height);
		assert(value.format == View::Gray8 && background.format == View::Gray8);

		EdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, background.data, background.stride);
	}

	void EdgeBackgroundShiftRange(const View & value, View & background, const View & mask)
	{
		assert(value.width == background.width && value.height == background.height && 
			value.width == mask.width && value.height == mask.height);
		assert(value.format == View::Gray8 && background.format == View::Gray8 && mask.format == View::Gray8);

		EdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, 
			background.data, background.stride, mask.data, mask.stride);
	}
}