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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, __mmask64 m = -1)
        {
            const __m512i _value = Load<align, mask>(value, m);
            const __m512i _lo = Load<align, mask>(lo, m);
            const __m512i _hi = Load<align, mask>(hi, m);

            const __mmask64 inc = _mm512_cmpgt_epu8_mask(_value, _hi);
            const __mmask64 dec = _mm512_cmplt_epu8_mask(_value, _lo);

            Store<align, mask>(lo, _mm512_mask_subs_epu8(_lo, dec, _lo, K8_01), m);
            Store<align, mask>(hi, _mm512_mask_adds_epu8(_hi, inc, _hi, K8_01), m);
        }

        template <bool align> void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(lo) && Aligned(loStride));
                assert(Aligned(hi) && Aligned(hiStride));
            }

			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
                for(; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow<align, false>(value + col, lo + col, hi + col);
                if(col < width)
					BackgroundGrowRangeSlow<align, true>(value + col, lo + col, hi + col, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                BackgroundGrowRangeSlow<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
            else
                BackgroundGrowRangeSlow<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
        }

		template <bool align, bool mask> SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t * value, uint8_t * lo, uint8_t * hi, __mmask64 m = -1)
		{
			const __m512i _value = Load<align, mask>(value, m);
			const __m512i _lo = Load<align, mask>(lo, m);
			const __m512i _hi = Load<align, mask>(hi, m);

			Store<align, mask>(lo, _mm512_min_epu8(_lo, _value), m);
			Store<align, mask>(hi, _mm512_max_epu8(_hi, _value), m);
		}

		template <bool align> void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
		{
			if (align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(lo) && Aligned(loStride));
				assert(Aligned(hi) && Aligned(hiStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < alignedWidth; col += A)
					BackgroundGrowRangeFast<align, false>(value + col, lo + col, hi + col);
				if (col < width)
					BackgroundGrowRangeFast<align, true>(value + col, lo + col, hi + col, tailMask);
				value += valueStride;
				lo += loStride;
				hi += hiStride;
			}
		}

		void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
		{
			if (Aligned(value) && Aligned(valueStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
				BackgroundGrowRangeFast<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
			else
				BackgroundGrowRangeFast<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
		}

		template <bool align, bool mask> SIMD_INLINE void BackgroundIncrementCount(const uint8_t * value,
			const uint8_t * loValue, const uint8_t * hiValue, uint8_t * loCount, uint8_t * hiCount, size_t offset, __mmask64 m = -1)
		{
			const __m512i _value = Load<align, mask>(value + offset, m);
			const __m512i _loValue = Load<align, mask>(loValue + offset, m);
			const __m512i _loCount = Load<align, mask>(loCount + offset, m);
			const __m512i _hiValue = Load<align, mask>(hiValue + offset, m);
			const __m512i _hiCount = Load<align, mask>(hiCount + offset, m);

			const __mmask64 incLo = _mm512_cmplt_epu8_mask(_value, _loValue);
			const __mmask64 incHi = _mm512_cmpgt_epu8_mask(_value, _hiValue);

			Store<align, mask>(loCount + offset, _mm512_mask_adds_epu8(_loCount, incLo, _loCount, K8_01), m);
			Store<align, mask>(hiCount + offset, _mm512_mask_adds_epu8(_hiCount, incHi, _hiCount, K8_01), m);
		}

		template <bool align> void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
			uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
		{
			if (align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride));
				assert(Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			__mmask64 tailMask = TailMask64(width - alignedWidth);
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < alignedWidth; col += A)
					BackgroundIncrementCount<align, false>(value, loValue, hiValue, loCount, hiCount, col);
				if (col < width)
					BackgroundIncrementCount<align, true>(value, loValue, hiValue, loCount, hiCount, col, tailMask);
				value += valueStride;
				loValue += loValueStride;
				hiValue += hiValueStride;
				loCount += loCountStride;
				hiCount += hiCountStride;
			}
		}

		void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
			uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
		{
			if (Aligned(value) && Aligned(valueStride) &&
				Aligned(loValue) && Aligned(loValueStride) && Aligned(hiValue) && Aligned(hiValueStride) &&
				Aligned(loCount) && Aligned(loCountStride) && Aligned(hiCount) && Aligned(hiCountStride))
				BackgroundIncrementCount<true>(value, valueStride, width, height,
					loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
			else
				BackgroundIncrementCount<false>(value, valueStride, width, height,
					loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
		}
    }
#endif// SIMD_AVX512BW_ENABLE
}
