/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdSet.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void BackgroundGrowRangeSlow(const uint8_t * value, uint8_t * lo, uint8_t * hi, __mmask64 m = -1)
        {
            const __m512i _value = mask ? _mm512_maskz_loadu_epi8(m, value) : _mm512_loadu_si512(value);
            const __m512i _lo = mask ? _mm512_maskz_loadu_epi8(m, lo) : _mm512_loadu_si512(lo);
            const __m512i _hi = mask ? _mm512_maskz_loadu_epi8(m, hi) : _mm512_loadu_si512(hi);

            const __mmask64 inc = _mm512_cmpgt_epu8_mask(_value, _hi);
            const __mmask64 dec = _mm512_cmplt_epu8_mask(_value, _lo);

            const __m512i loOut = _mm512_mask_subs_epu8(_lo, dec, _lo, K8_01);
            const __m512i hiOut = _mm512_mask_adds_epu8(_hi, inc, _hi, K8_01);
            if (mask)
            {
                _mm512_mask_storeu_epi8(lo, m, loOut);
                _mm512_mask_storeu_epi8(hi, m, hiOut);
            }
            else
            {
                _mm512_storeu_si512(lo, loOut);
                _mm512_storeu_si512(hi, hiOut);
            }
        }

        void BackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundGrowRangeSlow<false>(value + col, lo + col, hi + col);
                if (col < width)
                    BackgroundGrowRangeSlow<true>(value + col, lo + col, hi + col, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t * value, uint8_t * lo, uint8_t * hi, __mmask64 m = -1)
        {
            const __m512i _value = mask ? _mm512_maskz_loadu_epi8(m, value) : _mm512_loadu_si512(value);
            const __m512i _lo = mask ? _mm512_maskz_loadu_epi8(m, lo) : _mm512_loadu_si512(lo);
            const __m512i _hi = mask ? _mm512_maskz_loadu_epi8(m, hi) : _mm512_loadu_si512(hi);

            const __m512i loOut = _mm512_min_epu8(_lo, _value);
            const __m512i hiOut = _mm512_max_epu8(_hi, _value);
            if (mask)
            {
                _mm512_mask_storeu_epi8(lo, m, loOut);
                _mm512_mask_storeu_epi8(hi, m, hiOut);
            }
            else
            {
                _mm512_storeu_si512(lo, loOut);
                _mm512_storeu_si512(hi, hiOut);
            }
        }

        void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundGrowRangeFast<false>(value + col, lo + col, hi + col);
                if (col < width)
                    BackgroundGrowRangeFast<true>(value + col, lo + col, hi + col, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundIncrementCount(const uint8_t * value,
            const uint8_t * loValue, const uint8_t * hiValue, uint8_t * loCount, uint8_t * hiCount, size_t offset, __mmask64 m = -1)
        {
            const __m512i _value = mask ? _mm512_maskz_loadu_epi8(m, value + offset) : _mm512_loadu_si512(value + offset);
            const __m512i _loValue = mask ? _mm512_maskz_loadu_epi8(m, loValue + offset) : _mm512_loadu_si512(loValue + offset);
            const __m512i _loCount = mask ? _mm512_maskz_loadu_epi8(m, loCount + offset) : _mm512_loadu_si512(loCount + offset);
            const __m512i _hiValue = mask ? _mm512_maskz_loadu_epi8(m, hiValue + offset) : _mm512_loadu_si512(hiValue + offset);
            const __m512i _hiCount = mask ? _mm512_maskz_loadu_epi8(m, hiCount + offset) : _mm512_loadu_si512(hiCount + offset);

            const __mmask64 incLo = _mm512_cmplt_epu8_mask(_value, _loValue);
            const __mmask64 incHi = _mm512_cmpgt_epu8_mask(_value, _hiValue);

            const __m512i loOut = _mm512_mask_adds_epu8(_loCount, incLo, _loCount, K8_01);
            const __m512i hiOut = _mm512_mask_adds_epu8(_hiCount, incHi, _hiCount, K8_01);
            if (mask)
            {
                _mm512_mask_storeu_epi8(loCount + offset, m, loOut);
                _mm512_mask_storeu_epi8(hiCount + offset, m, hiOut);
            }
            else
            {
                _mm512_storeu_si512(loCount + offset, loOut);
                _mm512_storeu_si512(hiCount + offset, hiOut);
            }
        }

        void BackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
            uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundIncrementCount<false>(value, loValue, hiValue, loCount, hiCount, col);
                if (col < width)
                    BackgroundIncrementCount<true>(value, loValue, hiValue, loCount, hiCount, col, tailMask);
                value += valueStride;
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        SIMD_INLINE __m512i AdjustLo(const __m512i & count, const __m512i & value, const __m512i & threshold)
        {
            const __mmask64 dec = _mm512_cmpgt_epu8_mask(count, threshold);
            const __mmask64 inc = _mm512_cmplt_epu8_mask(count, threshold);
            __m512i added = _mm512_mask_adds_epu8(value, inc, value, K8_01);
            return _mm512_mask_subs_epu8(added, dec, added, K8_01);
        }

        SIMD_INLINE __m512i AdjustHi(const __m512i & count, const __m512i & value, const __m512i & threshold)
        {
            const __mmask64 inc = _mm512_cmpgt_epu8_mask(count, threshold);
            const __mmask64 dec = _mm512_cmplt_epu8_mask(count, threshold);
            __m512i added = _mm512_mask_adds_epu8(value, inc, value, K8_01);
            return _mm512_mask_subs_epu8(added, dec, added, K8_01);
        }

        template <bool mask> SIMD_INLINE void BackgroundAdjustRange(uint8_t * loCount, uint8_t * loValue,
            uint8_t * hiCount, uint8_t * hiValue, const __m512i & threshold, __mmask64 m = -1)
        {
            const __m512i _loCount = mask ? _mm512_maskz_loadu_epi8(m, loCount) : _mm512_loadu_si512(loCount);
            const __m512i _loValue = mask ? _mm512_maskz_loadu_epi8(m, loValue) : _mm512_loadu_si512(loValue);
            const __m512i _hiCount = mask ? _mm512_maskz_loadu_epi8(m, hiCount) : _mm512_loadu_si512(hiCount);
            const __m512i _hiValue = mask ? _mm512_maskz_loadu_epi8(m, hiValue) : _mm512_loadu_si512(hiValue);

            const __m512i loValueOut = AdjustLo(_loCount, _loValue, threshold);
            const __m512i hiValueOut = AdjustHi(_hiCount, _hiValue, threshold);
            if (mask)
            {
                _mm512_mask_storeu_epi8(loValue, m, loValueOut);
                _mm512_mask_storeu_epi8(hiValue, m, hiValueOut);
                _mm512_mask_storeu_epi8(loCount, m, K_ZERO);
                _mm512_mask_storeu_epi8(hiCount, m, K_ZERO);
            }
            else
            {
                _mm512_storeu_si512(loValue, loValueOut);
                _mm512_storeu_si512(hiValue, hiValueOut);
                _mm512_storeu_si512(loCount, K_ZERO);
                _mm512_storeu_si512(hiCount, K_ZERO);
            }
        }

        void BackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
        {
            const __m512i _threshold = _mm512_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundAdjustRange<false>(loCount + col, loValue + col, hiCount + col, hiValue + col, _threshold);
                if (col < width)
                    BackgroundAdjustRange<true>(loCount + col, loValue + col, hiCount + col, hiValue + col, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundAdjustRangeMasked(uint8_t * loCount, uint8_t * loValue, uint8_t * hiCount, uint8_t * hiValue,
            const uint8_t * pmask, const __m512i & threshold, __mmask64 m = -1)
        {
            const __m512i _mask = mask ? _mm512_maskz_loadu_epi8(m, pmask) : _mm512_loadu_si512(pmask);
            const __mmask64 mm = _mm512_cmpneq_epu8_mask(_mask, K_ZERO) & m;

            const __m512i _loCount = mask ? _mm512_maskz_loadu_epi8(m, loCount) : _mm512_loadu_si512(loCount);
            const __m512i _loValue = mask ? _mm512_maskz_loadu_epi8(m, loValue) : _mm512_loadu_si512(loValue);
            const __m512i _hiCount = mask ? _mm512_maskz_loadu_epi8(m, hiCount) : _mm512_loadu_si512(hiCount);
            const __m512i _hiValue = mask ? _mm512_maskz_loadu_epi8(m, hiValue) : _mm512_loadu_si512(hiValue);

            _mm512_mask_storeu_epi8(loValue, mm, AdjustLo(_loCount, _loValue, threshold));
            _mm512_mask_storeu_epi8(hiValue, mm, AdjustHi(_hiCount, _hiValue, threshold));
            if (mask)
            {
                _mm512_mask_storeu_epi8(loCount, m, K_ZERO);
                _mm512_mask_storeu_epi8(hiCount, m, K_ZERO);
            }
            else
            {
                _mm512_storeu_si512(loCount, K_ZERO);
                _mm512_storeu_si512(hiCount, K_ZERO);
            }
        }

        void BackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
            uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
        {
            const __m512i _threshold = _mm512_set1_epi8((char)threshold);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundAdjustRangeMasked<false>(loCount + col, loValue + col, hiCount + col, hiValue + col, mask + col, _threshold);
                if (col < width)
                    BackgroundAdjustRangeMasked<true>(loCount + col, loValue + col, hiCount + col, hiValue + col, mask + col, _threshold, tailMask);
                loValue += loValueStride;
                hiValue += hiValueStride;
                loCount += loCountStride;
                hiCount += hiCountStride;
                mask += maskStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundShiftRange(const uint8_t * value, uint8_t * lo, uint8_t * hi, __mmask64 m = -1)
        {
            const __m512i _value = mask ? _mm512_maskz_loadu_epi8(m, value) : _mm512_loadu_si512(value);
            const __m512i _lo = mask ? _mm512_maskz_loadu_epi8(m, lo) : _mm512_loadu_si512(lo);
            const __m512i _hi = mask ? _mm512_maskz_loadu_epi8(m, hi) : _mm512_loadu_si512(hi);

            const __m512i add = _mm512_subs_epu8(_value, _hi);
            const __m512i sub = _mm512_subs_epu8(_lo, _value);

            const __m512i loOut = _mm512_subs_epu8(_mm512_adds_epu8(_lo, add), sub);
            const __m512i hiOut = _mm512_subs_epu8(_mm512_adds_epu8(_hi, add), sub);
            if (mask)
            {
                _mm512_mask_storeu_epi8(lo, m, loOut);
                _mm512_mask_storeu_epi8(hi, m, hiOut);
            }
            else
            {
                _mm512_storeu_si512(lo, loOut);
                _mm512_storeu_si512(hi, hiOut);
            }
        }

        void BackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundShiftRange<false>(value + col, lo + col, hi + col);
                if (col < width)
                    BackgroundShiftRange<true>(value + col, lo + col, hi + col, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundShiftRangeMasked(const uint8_t * value, uint8_t * lo, uint8_t * hi,
            const uint8_t * pmask, __mmask64 m = -1)
        {
            const __m512i _mask = mask ? _mm512_maskz_loadu_epi8(m, pmask) : _mm512_loadu_si512(pmask);
            const __mmask64 mm = _mm512_cmpneq_epu8_mask(_mask, K_ZERO) & m;

            const __m512i _value = mask ? _mm512_maskz_loadu_epi8(m, value) : _mm512_loadu_si512(value);
            const __m512i _lo = mask ? _mm512_maskz_loadu_epi8(m, lo) : _mm512_loadu_si512(lo);
            const __m512i _hi = mask ? _mm512_maskz_loadu_epi8(m, hi) : _mm512_loadu_si512(hi);

            const __m512i add = _mm512_subs_epu8(_value, _hi);
            const __m512i sub = _mm512_subs_epu8(_lo, _value);

            _mm512_mask_storeu_epi8(lo, mm, _mm512_subs_epu8(_mm512_adds_epu8(_lo, add), sub));
            _mm512_mask_storeu_epi8(hi, mm, _mm512_subs_epu8(_mm512_adds_epu8(_hi, add), sub));
        }

        void BackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
            uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundShiftRangeMasked<false>(value + col, lo + col, hi + col, mask + col);
                if (col < width)
                    BackgroundShiftRangeMasked<true>(value + col, lo + col, hi + col, mask + col, tailMask);
                value += valueStride;
                lo += loStride;
                hi += hiStride;
                mask += maskStride;
            }
        }

        template <bool mask> SIMD_INLINE void BackgroundInitMask(const uint8_t * src, uint8_t * dst, const __m512i & index, const __m512i & value, __mmask64 m = -1)
        {
            __m512i _src = mask ? _mm512_maskz_loadu_epi8(m, src) : _mm512_loadu_si512(src);
            __mmask64 mm = _mm512_cmpeq_epu8_mask(_src, index) & m;
            _mm512_mask_storeu_epi8(dst, mm, value);
        }

        void BackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i _index = _mm512_set1_epi8(index);
            __m512i _value = _mm512_set1_epi8(value);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BackgroundInitMask<false>(src + col, dst + col, _index, _value);
                if (col < width)
                    BackgroundInitMask<true>(src + col, dst + col, _index, _value, tailMask);
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
