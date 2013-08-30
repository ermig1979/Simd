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

        void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            for(size_t row = 0; row < height; ++row)
            {
                uint sum = 0;
                for(size_t col = 0; col < width; ++col)
                    sum += src[col];
                sums[row] = sum;
                src += stride;
            }
        }

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            memset(sums, 0, sizeof(uint)*width);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                    sums[col] += src[col];
                src += stride;
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

        template <bool align> void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint)*height);
            for(size_t row = 0; row < height; ++row)
            {
                __m128i sum = _mm_setzero_si128();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _src = Load<align>((__m128i*)(src + col));
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src, K_ZERO));
                }
                if(alignedWidth != width)
                {
                    __m128i _src = _mm_and_si128(Load<false>((__m128i*)(src + width - A)), tailMask);
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src, K_ZERO));
                }
                sums[row] = ExtractInt32Sum(sum);
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

        template <bool align> SIMD_INLINE void Sum16(__m128i src8, ushort * sums16)
        {
            Store<align>((__m128i*)sums16 + 0, _mm_add_epi16(Load<align>((__m128i*)sums16 + 0), _mm_unpacklo_epi8(src8, K_ZERO)));
            Store<align>((__m128i*)sums16 + 1, _mm_add_epi16(Load<align>((__m128i*)sums16 + 1), _mm_unpackhi_epi8(src8, K_ZERO)));
        }

        template <bool align> SIMD_INLINE void Sum32(__m128i src16, uint * sums32)
        {
            Store<align>((__m128i*)sums32 + 0, _mm_add_epi32(Load<align>((__m128i*)sums32 + 0), _mm_unpacklo_epi16(src16, K_ZERO)));
            Store<align>((__m128i*)sums32 + 1, _mm_add_epi32(Load<align>((__m128i*)sums32 + 1), _mm_unpackhi_epi16(src16, K_ZERO)));
        }

        template <bool align> void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint)*alignedHiWidth);
            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(ushort)*width);
                for(size_t row = rowStart; row < rowEnd; ++row)
                {
                    for(size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m128i src8 = Load<align>((__m128i*)(src + col));
                        Sum16<true>(src8, buffer.sums16 + col);
                    }
                    if(alignedLoWidth != width)
                    {
                        __m128i src8 = _mm_and_si128(Load<false>((__m128i*)(src + width - A)), tailMask);
                        Sum16<false>(src8, buffer.sums16 + width - A);
                    }
                    src += stride;
                }

                for(size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    __m128i src16 = Load<true>((__m128i*)(buffer.sums16 + col));
                    Sum32<true>(src16, buffer.sums32 + col);
                }
            }
            memcpy(sums, buffer.sums32, sizeof(uint)*width);
        }

        void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
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

    void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::GetRowSums(src, stride, width, height, sums);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::GetRowSums(src, stride, width, height, sums);
        else
#endif// SIMD_SSE2_ENABLE
            Base::GetRowSums(src, stride, width, height, sums);
    }

    void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::GetColSums(src, stride, width, height, sums);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::GetColSums(src, stride, width, height, sums);
        else
#endif// SIMD_SSE2_ENABLE
            Base::GetColSums(src, stride, width, height, sums);
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

    void GetRowSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    void GetColSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetColSums(src.data, src.stride, src.width, src.height, sums);
    }
}