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
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		template <bool align> void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			assert(width*height && width >= A);
			if(align)
				assert(Aligned(src) && Aligned(stride));

			size_t bodyWidth = AlignLo(width, A);
			__m256i tailMask = SetMask<uchar>(0, A - width + bodyWidth, 0xFF);
			__m256i sum = _mm256_setzero_si256();
			__m256i min_ = K_INV_ZERO;
			__m256i max_ = K_ZERO;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m256i value = Load<align>((__m256i*)(src + col));
					min_ = _mm256_min_epu8(min_, value);
					max_ = _mm256_max_epu8(max_, value);
					sum = _mm256_add_epi64(_mm256_sad_epu8(value, K_ZERO), sum);
				}
				if(width - bodyWidth)
				{
					const __m256i value = Load<false>((__m256i*)(src + width - A));
					min_ = _mm256_min_epu8(min_, value);
					max_ = _mm256_max_epu8(max_, value);
					sum = _mm256_add_epi64(_mm256_sad_epu8(_mm256_and_si256(tailMask, value), K_ZERO), sum);
				}
				src += stride;
			}

			uchar min_buffer[A], max_buffer[A];
			_mm256_storeu_si256((__m256i*)min_buffer, min_);
			_mm256_storeu_si256((__m256i*)max_buffer, max_);
			*min = UCHAR_MAX;
			*max = 0;
			for (size_t i = 0; i < A; ++i)
			{
				*min = Base::MinU8(min_buffer[i], *min);
				*max = Base::MaxU8(max_buffer[i], *max);
			}
			*average = (uchar)((ExtractSum<uint64_t>(sum) + UCHAR_MAX/2)/(width*height));
		}

		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			if(Aligned(src) && Aligned(stride))
				GetStatistic<true>(src, stride, width, height, min, max, average);
			else
				GetStatistic<false>(src, stride, width, height, min, max, average);
		}

        SIMD_INLINE void GetMoments16(__m256i row, __m256i col, 
            __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            x = _mm256_add_epi32(x, _mm256_madd_epi16(col, K16_0001));
            y = _mm256_add_epi32(y, _mm256_madd_epi16(row, K16_0001));
            xx = _mm256_add_epi32(xx, _mm256_madd_epi16(col, col));
            xy = _mm256_add_epi32(xy, _mm256_madd_epi16(col, row));
            yy = _mm256_add_epi32(yy, _mm256_madd_epi16(row,row));
        }

        SIMD_INLINE void GetMoments8(__m256i mask, __m256i & row, __m256i & col, 
            __m256i & area, __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            area = _mm256_add_epi64(area, _mm256_sad_epu8(_mm256_and_si256(K8_01, mask), K_ZERO));

            const __m256i lo = _mm256_cmpeq_epi16(_mm256_unpacklo_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16(_mm256_and_si256(lo, row), _mm256_and_si256(lo, col), x, y, xx, xy, yy);
            col = _mm256_add_epi16(col, K16_0008);

            const __m256i hi = _mm256_cmpeq_epi16(_mm256_unpackhi_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16(_mm256_and_si256(hi, row), _mm256_and_si256(hi, col), x, y, xx, xy, yy);
            col = _mm256_add_epi16(col, K16_0018);
        }

        template <bool align> void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            assert(width >= A && width < SHRT_MAX && height < SHRT_MAX);
            if(align)
                assert(Aligned(mask) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uchar>(0, A - width + alignedWidth, 0xFF);

            const __m256i K16_I = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            const __m256i _index = _mm256_set1_epi8(index);
            const __m256i tailCol = _mm256_add_epi16(K16_I, _mm256_set1_epi16((ushort)(width - A)));

            __m256i _area = _mm256_setzero_si256();
            __m256i _x = _mm256_setzero_si256();
            __m256i _y = _mm256_setzero_si256();
            __m256i _xx = _mm256_setzero_si256();
            __m256i _xy = _mm256_setzero_si256();
            __m256i _yy = _mm256_setzero_si256();

            for(size_t row = 0; row < height; ++row)
            {
                __m256i _col = K16_I;
                __m256i _row = _mm256_set1_epi16((short)row);

                __m256i _rowX = _mm256_setzero_si256();
                __m256i _rowY = _mm256_setzero_si256();
                __m256i _rowXX = _mm256_setzero_si256();
                __m256i _rowXY = _mm256_setzero_si256();
                __m256i _rowYY = _mm256_setzero_si256();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _mask = _mm256_cmpeq_epi8(Load<align>((__m256i*)(mask + col)), _index);
                    GetMoments8(_mask, _row, _col, _area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                if(alignedWidth != width)
                {
                    __m256i _mask = _mm256_and_si256(_mm256_cmpeq_epi8(Load<false>((__m256i*)(mask + width - A)), _index), tailMask);
                    _col = tailCol;
                    GetMoments8(_mask, _row, _col, _area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                _x = _mm256_add_epi64(_x, HorizontalSum32(_rowX));
                _y = _mm256_add_epi64(_y, HorizontalSum32(_rowY));
                _xx = _mm256_add_epi64(_xx, HorizontalSum32(_rowXX));
                _xy = _mm256_add_epi64(_xy, HorizontalSum32(_rowXY));
                _yy = _mm256_add_epi64(_yy, HorizontalSum32(_rowYY));

                mask += stride;
            }
            *area = ExtractSum<int64_t>(_area);
            *x = ExtractSum<int64_t>(_x);
            *y = ExtractSum<int64_t>(_y);
            *xx = ExtractSum<int64_t>(_xx);
            *xy = ExtractSum<int64_t>(_xy);
            *yy = ExtractSum<int64_t>(_yy);
        }

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            if(Aligned(mask) && Aligned(stride))
                GetMoments<true>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
            else
                GetMoments<false>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        }

        template <bool align> void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uchar>(0, A - width + alignedWidth, 0xFF);

            memset(sums, 0, sizeof(uint)*height);
            for(size_t row = 0; row < height; ++row)
            {
                __m256i sum = _mm256_setzero_si256();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _src = Load<align>((__m256i*)(src + col));
                    sum = _mm256_add_epi32(sum, _mm256_sad_epu8(_src, K_ZERO));
                }
                if(alignedWidth != width)
                {
                    __m256i _src = _mm256_and_si256(Load<false>((__m256i*)(src + width - A)), tailMask);
                    sum = _mm256_add_epi32(sum, _mm256_sad_epu8(_src, K_ZERO));
                }
                sums[row] = ExtractSum<uint32_t>(sum);
                src += stride;
            }
        }

        void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(ushort)*width + sizeof(uint)*width);
                    sums16 = (ushort*)_p;
                    sums32 = (uint*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                ushort * sums16;
                uint * sums32;
            private:
                void *_p;
            };
        }

        SIMD_INLINE void Sum16(__m256i src8, ushort * sums16)
        {
            Store<true>((__m256i*)sums16 + 0, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 0), _mm256_unpacklo_epi8(src8, K_ZERO)));
            Store<true>((__m256i*)sums16 + 1, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 1), _mm256_unpackhi_epi8(src8, K_ZERO)));
        }

        SIMD_INLINE void Sum16To32(const ushort * src, uint * dst)
        {
            __m256i lo = LoadPermuted<true>((__m256i*)src + 0);
            __m256i hi = LoadPermuted<true>((__m256i*)src + 1);
            Store<true>((__m256i*)dst + 0, _mm256_add_epi32(Load<true>((__m256i*)dst + 0), _mm256_unpacklo_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 1, _mm256_add_epi32(Load<true>((__m256i*)dst + 1), _mm256_unpacklo_epi16(hi, K_ZERO)));
            Store<true>((__m256i*)dst + 2, _mm256_add_epi32(Load<true>((__m256i*)dst + 2), _mm256_unpackhi_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 3, _mm256_add_epi32(Load<true>((__m256i*)dst + 3), _mm256_unpackhi_epi16(hi, K_ZERO)));
        }

        template <bool align> void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint)*alignedHiWidth);
            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(ushort)*alignedHiWidth);
                for(size_t row = rowStart; row < rowEnd; ++row)
                {
                    for(size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m256i src8 = Load<align>((__m256i*)(src + col));
                        Sum16(src8, buffer.sums16 + col);
                    }
                    if(alignedLoWidth != width)
                    {
                        __m256i src8 = Load<false>((__m256i*)(src + width - A));
                        Sum16(src8, buffer.sums16 + alignedLoWidth);
                    }
                    src += stride;
                }

                for(size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint)*alignedLoWidth);
            if(alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint)*(width - alignedLoWidth));
        }

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDyRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uchar>(0, A - width + alignedWidth, 0xFF);

            memset(sums, 0, sizeof(uint)*height);
            const uchar * src0 = src;
            const uchar * src1 = src + stride;
            height--;
            for(size_t row = 0; row < height; ++row)
            {
                __m256i sum = _mm256_setzero_si256();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _src0 = Load<align>((__m256i*)(src0 + col));
                    __m256i _src1 = Load<align>((__m256i*)(src1 + col));
                    sum = _mm256_add_epi32(sum, _mm256_sad_epu8(_src0, _src1));
                }
                if(alignedWidth != width)
                {
                    __m256i _src0 = _mm256_and_si256(Load<false>((__m256i*)(src0 + width - A)), tailMask);
                    __m256i _src1 = _mm256_and_si256(Load<false>((__m256i*)(src1 + width - A)), tailMask);
                    sum = _mm256_add_epi32(sum, _mm256_sad_epu8(_src0, _src1));
                }
                sums[row] = ExtractSum<uint32_t>(sum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDxColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint)*alignedHiWidth);
            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(ushort)*alignedHiWidth);
                for(size_t row = rowStart; row < rowEnd; ++row)
                {
                    for(size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m256i _src0 = Load<align>((__m256i*)(src + col + 0));
                        __m256i _src1 = Load<false>((__m256i*)(src + col + 1));
                        Sum16(AbsDifferenceU8(_src0, _src1), buffer.sums16 + col);
                    }
                    if(alignedLoWidth != width)
                    {
                        __m256i _src0 = Load<false>((__m256i*)(src + width - A + 0));
                        __m256i _src1 = Load<false>((__m256i*)(src + width - A + 1));
                        Sum16(AbsDifferenceU8(_src0, _src1), buffer.sums16 + alignedLoWidth);
                    }
                    src += stride;
                }

                for(size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint)*alignedLoWidth);
            if(alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint)*(width - alignedLoWidth));
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        template <bool align, SimdCompareType compareType> 
        void ConditionalCount(const uchar * src, size_t stride, size_t width, size_t height, uchar value, uint * count)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            __m256i tailMask = SetMask<uchar>(0, A - width + alignedWidth, 0xFF);;

            __m256i _value = _mm256_set1_epi8(value);
            __m256i _count = _mm256_setzero_si256();
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m256i mask = Compare<compareType>(Load<align>((__m256i*)(src + col)), _value);
                    _count = _mm256_add_epi64(_count, _mm256_sad_epu8(_mm256_and_si256(mask, K8_01), K_ZERO));
                }
                if(alignedWidth != width)
                {
                    const __m256i mask = _mm256_and_si256(Compare<compareType>(Load<false>((__m256i*)(src + width - A)), _value), tailMask);
                    _count = _mm256_add_epi64(_count, _mm256_sad_epu8(_mm256_and_si256(mask, K8_01), K_ZERO));
                }
                src += stride;
            }
            *count = ExtractSum<uint32_t>(_count);
        }

        template <SimdCompareType compareType> 
        void ConditionalCount(const uchar * src, size_t stride, size_t width, size_t height, uchar value, uint * count)
        {
            if(Aligned(src) && Aligned(stride))
                ConditionalCount<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount(const uchar * src, size_t stride, size_t width, size_t height, 
            uchar value, SimdCompareType compareType, uint * count)
        {
            switch(compareType)
            {
            case SimdCompareEqual: 
                return ConditionalCount<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual: 
                return ConditionalCount<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater: 
                return ConditionalCount<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual: 
                return ConditionalCount<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser: 
                return ConditionalCount<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual: 
                return ConditionalCount<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default: 
                assert(0);
            }
        }

        template <bool align, SimdCompareType compareType> 
        void ConditionalSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * mask, size_t maskStride, uchar value, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            __m256i tailMask = SetMask<uchar>(0, A - width + alignedWidth, 0xFF);;

            __m256i _value = _mm256_set1_epi8(value);
            __m256i _sum = _mm256_setzero_si256();
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m256i _src = Load<align>((__m256i*)(src + col));
                    const __m256i _mask = Compare<compareType>(Load<align>((__m256i*)(mask + col)), _value);
                    _sum = _mm256_add_epi64(_sum, _mm256_sad_epu8(_mm256_and_si256(_mask, _src), K_ZERO));
                }
                if(alignedWidth != width)
                {
                    const __m256i _src = Load<false>((__m256i*)(src + width - A));
                    const __m256i _mask = _mm256_and_si256(Compare<compareType>(Load<false>((__m256i*)(mask + width - A)), _value), tailMask);
                    _sum = _mm256_add_epi64(_sum, _mm256_sad_epu8(_mm256_and_si256(_mask, _src), K_ZERO));
                }
                src += srcStride;
                mask += maskStride;
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        template <SimdCompareType compareType> 
        void ConditionalSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * mask, size_t maskStride, uchar value, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
            const uchar * mask, size_t maskStride, uchar value, SimdCompareType compareType, uint64_t * sum)
        {
            switch(compareType)
            {
            case SimdCompareEqual: 
                return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual: 
                return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater: 
                return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual: 
                return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser: 
                return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual: 
                return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default: 
                assert(0);
            }
        }
	}
#endif// SIMD_AVX2_ENABLE
}