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
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdStatistic.h"

namespace Simd
{
	namespace Base
	{
		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			assert(width*height);

			uint64_t sum = 0;
			int min_ = UCHAR_MAX;
			int max_ = 0;
			for(size_t row = 0; row < height; ++row)
			{
				int rowSum = 0;
				for(size_t col = 0; col < width; ++col)
				{
					int value = src[col];
					max_ = MaxU8(value, max_);
					min_ = MinU8(value, min_);
					rowSum += value;
				}
				sum += rowSum;
				src += stride;
			}
			*average = (uchar)((sum + UCHAR_MAX/2)/(width*height));
			*min = min_;
			*max = max_;
		}

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            *area = 0;
            *x = 0;
            *y = 0;
            *xx = 0;
            *xy = 0;
            *yy = 0;
            for(size_t row = 0; row < height; ++row)
            {
                size_t rowArea = 0;
                size_t rowX = 0;
                size_t rowY = 0;
                size_t rowXX = 0;
                size_t rowXY = 0;
                size_t rowYY = 0;
                for(size_t col = 0; col < width; ++col)
                {
                    if(mask[col] == index)
                    {
                        rowArea++;
                        rowX += col;
                        rowY += row;
                        rowXX += col*col;
                        rowXY += col*row;
                        rowYY += row*row;
                    }               
                }
                *area += rowArea;
                *x += rowX;
                *y += rowY;
                *xx += rowXX;
                *xy += rowXY;
                *yy += rowYY;

                mask += stride;
            }
        }
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		template <bool align> void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			assert(width*height && width >= A);
			if(align)
				assert(Aligned(src) && Aligned(stride));

			size_t bodyWidth = AlignLo(width, A);
			__m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
			__m128i sum = _mm_setzero_si128();
			__m128i min_ = K_INV_ZERO;
			__m128i max_ = K_ZERO;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < bodyWidth; col += A)
				{
					const __m128i value = Load<align>((__m128i*)(src + col));
					min_ = _mm_min_epu8(min_, value);
					max_ = _mm_max_epu8(max_, value);
					sum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), sum);
				}
				if(width - bodyWidth)
				{
					const __m128i value = Load<false>((__m128i*)(src + width - A));
					min_ = _mm_min_epu8(min_, value);
					max_ = _mm_max_epu8(max_, value);
					sum = _mm_add_epi64(_mm_sad_epu8(_mm_and_si128(tailMask, value), K_ZERO), sum);
				}
				src += stride;
			}

			uchar min_buffer[A], max_buffer[A];
			_mm_storeu_si128((__m128i*)min_buffer, min_);
			_mm_storeu_si128((__m128i*)max_buffer, max_);
			*min = UCHAR_MAX;
			*max = 0;
			for (size_t i = 0; i < A; ++i)
			{
				*min = Base::MinU8(min_buffer[i], *min);
				*max = Base::MaxU8(max_buffer[i], *max);
			}
			*average = (uchar)((ExtractInt64Sum(sum) + UCHAR_MAX/2)/(width*height));
		}

		void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
			uchar * min, uchar * max, uchar * average)
		{
			if(Aligned(src) && Aligned(stride))
				GetStatistic<true>(src, stride, width, height, min, max, average);
			else
				GetStatistic<false>(src, stride, width, height, min, max, average);
		}

        SIMD_INLINE void GetMoments16(__m128i row, __m128i col, 
            __m128i & x, __m128i & y, __m128i & xx, __m128i & xy, __m128i & yy)
        {
            x = _mm_add_epi32(x, _mm_madd_epi16(col, K16_0001));
            y = _mm_add_epi32(y, _mm_madd_epi16(row, K16_0001));
            xx = _mm_add_epi32(xx, _mm_madd_epi16(col, col));
            xy = _mm_add_epi32(xy, _mm_madd_epi16(col, row));
            yy = _mm_add_epi32(yy, _mm_madd_epi16(row,row));
        }

        SIMD_INLINE void GetMoments8(__m128i mask, __m128i & row, __m128i & col, 
            __m128i & area, __m128i & x, __m128i & y, __m128i & xx, __m128i & xy, __m128i & yy)
        {
            area = _mm_add_epi64(area, _mm_sad_epu8(_mm_and_si128(K8_01, mask), K_ZERO));

            const __m128i lo = _mm_cmpeq_epi16(_mm_unpacklo_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16(_mm_and_si128(lo, row), _mm_and_si128(lo, col), x, y, xx, xy, yy);
            col = _mm_add_epi16(col, K16_0008);

            const __m128i hi = _mm_cmpeq_epi16(_mm_unpackhi_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16(_mm_and_si128(hi, row), _mm_and_si128(hi, col), x, y, xx, xy, yy);
            col = _mm_add_epi16(col, K16_0008);
        }

        template <bool align> void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            assert(width >= A && width < SHRT_MAX && height < SHRT_MAX);
            if(align)
                assert(Aligned(mask) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);

            const __m128i K16_I = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
            const __m128i _index = _mm_set1_epi8(index);
            const __m128i tailCol = _mm_add_epi16(K16_I, _mm_set1_epi16((ushort)(width - A)));

            __m128i _area = _mm_setzero_si128();
            __m128i _x = _mm_setzero_si128();
            __m128i _y = _mm_setzero_si128();
            __m128i _xx = _mm_setzero_si128();
            __m128i _xy = _mm_setzero_si128();
            __m128i _yy = _mm_setzero_si128();

            for(size_t row = 0; row < height; ++row)
            {
                __m128i _col = K16_I;
                __m128i _row = _mm_set1_epi16((short)row);

                __m128i _rowX = _mm_setzero_si128();
                __m128i _rowY = _mm_setzero_si128();
                __m128i _rowXX = _mm_setzero_si128();
                __m128i _rowXY = _mm_setzero_si128();
                __m128i _rowYY = _mm_setzero_si128();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _mask = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask + col)), _index);
                    GetMoments8(_mask, _row, _col, _area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                if(alignedWidth != width)
                {
                    __m128i _mask = _mm_and_si128(_mm_cmpeq_epi8(Load<false>((__m128i*)(mask + width - A)), _index), tailMask);
                    _col = tailCol;
                    GetMoments8(_mask, _row, _col, _area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                _x = _mm_add_epi64(_x, HorizontalSum32(_rowX));
                _y = _mm_add_epi64(_y, HorizontalSum32(_rowY));
                _xx = _mm_add_epi64(_xx, HorizontalSum32(_rowXX));
                _xy = _mm_add_epi64(_xy, HorizontalSum32(_rowXY));
                _yy = _mm_add_epi64(_yy, HorizontalSum32(_rowYY));

                mask += stride;
            }
            *area = ExtractInt64Sum(_area);
            *x = ExtractInt64Sum(_x);
            *y = ExtractInt64Sum(_y);
            *xx = ExtractInt64Sum(_xx);
            *xy = ExtractInt64Sum(_xy);
            *yy = ExtractInt64Sum(_yy);
       }

        void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            if(Aligned(mask) && Aligned(stride))
                GetMoments<true>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
            else
                GetMoments<false>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        }
	}
#endif// SIMD_SSE2_ENABLE

	void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
		uchar * min, uchar * max, uchar * average)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::GetStatistic(src, stride, width, height, min, max, average);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::GetStatistic(src, stride, width, height, min, max, average);
		else
#endif// SIMD_SSE2_ENABLE
			Base::GetStatistic(src, stride, width, height, min, max, average);
	}

    void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
        uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A && width < SHRT_MAX && height < SHRT_MAX)
            Avx2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A && width < SHRT_MAX && height < SHRT_MAX)
            Sse2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        else
#endif// SIMD_SSE2_ENABLE
            Base::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);    
    }

	void GetStatistic(const View & src, uchar * min, uchar * max, uchar * average)
	{
		assert(src.format == View::Gray8);

		GetStatistic(src.data, src.stride, src.width, src.height, min, max, average);
	}

    void GetMoments(const View & mask, uchar index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
    {
        assert(mask.format == View::Gray8);

        GetMoments(mask.data, mask.stride, mask.width, mask.height, index, area, x, y, xx, xy, yy);
    }
}