/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
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

            uint8_t min_buffer[A], max_buffer[A];
            _mm256_storeu_si256((__m256i*)min_buffer, min_);
            _mm256_storeu_si256((__m256i*)max_buffer, max_);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum<uint64_t>(sum) + width*height/2)/(width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if(Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

        SIMD_INLINE void GetMoments16Small(__m256i row, __m256i col, 
            __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            x = _mm256_add_epi32(x, _mm256_madd_epi16(col, K16_0001));
            y = _mm256_add_epi32(y, _mm256_madd_epi16(row, K16_0001));
            xx = _mm256_add_epi32(xx, _mm256_madd_epi16(col, col));
            xy = _mm256_add_epi32(xy, _mm256_madd_epi16(col, row));
            yy = _mm256_add_epi32(yy, _mm256_madd_epi16(row, row));
        }

        SIMD_INLINE void GetMoments16Large(__m256i row, __m256i col, 
            __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            x = _mm256_add_epi32(x, _mm256_madd_epi16(col, K16_0001));
            y = _mm256_add_epi32(y, _mm256_madd_epi16(row, K16_0001));
            xx = _mm256_madd_epi16(col, col);
            xy = _mm256_madd_epi16(col, row);
            yy = _mm256_madd_epi16(row, row);
        }

        SIMD_INLINE void GetMoments8Small(__m256i mask, __m256i & row, __m256i & col, 
            __m256i & area, __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            area = _mm256_add_epi64(area, _mm256_sad_epu8(_mm256_and_si256(K8_01, mask), K_ZERO));

            const __m256i lo = _mm256_cmpeq_epi16(_mm256_unpacklo_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16Small(_mm256_and_si256(lo, row), _mm256_and_si256(lo, col), x, y, xx, xy, yy);
            col = _mm256_add_epi16(col, K16_0008);

            const __m256i hi = _mm256_cmpeq_epi16(_mm256_unpackhi_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16Small(_mm256_and_si256(hi, row), _mm256_and_si256(hi, col), x, y, xx, xy, yy);
            col = _mm256_add_epi16(col, K16_0018);
        }

        SIMD_INLINE void GetMoments8Large(__m256i mask, __m256i & row, __m256i & col, 
            __m256i & area, __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            area = _mm256_add_epi64(area, _mm256_sad_epu8(_mm256_and_si256(K8_01, mask), K_ZERO));

            __m256i xxLo, xyLo, yyLo, maskLo = _mm256_cmpeq_epi16(_mm256_unpacklo_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16Large(_mm256_and_si256(maskLo, row), _mm256_and_si256(maskLo, col), x, y, xxLo, xyLo, yyLo);
            col = _mm256_add_epi16(col, K16_0008);

            __m256i xxHi, xyHi, yyHi, maskHi = _mm256_cmpeq_epi16(_mm256_unpackhi_epi8(mask, K_ZERO), K16_00FF);
            GetMoments16Large(_mm256_and_si256(maskHi, row), _mm256_and_si256(maskHi, col), x, y, xxHi, xyHi, yyHi);
            col = _mm256_add_epi16(col, K16_0018);

            xx = _mm256_add_epi64(xx, HorizontalSum32(_mm256_hadd_epi32(xxLo, xxHi)));
            xy = _mm256_add_epi64(xy, HorizontalSum32(_mm256_hadd_epi32(xyLo, xyHi)));
            yy = _mm256_add_epi64(yy, HorizontalSum32(_mm256_hadd_epi32(yyLo, yyHi)));
        }

        template <bool align> void GetMomentsSmall(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            __m256i & area, __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);

            const __m256i K16_I = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            const __m256i _index = _mm256_set1_epi8(index);
            const __m256i tailCol = _mm256_add_epi16(K16_I, _mm256_set1_epi16((uint16_t)(width - A)));

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
                    GetMoments8Small(_mask, _row, _col, area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                if(alignedWidth != width)
                {
                    __m256i _mask = _mm256_and_si256(_mm256_cmpeq_epi8(Load<false>((__m256i*)(mask + width - A)), _index), tailMask);
                    _col = tailCol;
                    GetMoments8Small(_mask, _row, _col, area, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
                }
                x = _mm256_add_epi64(x, HorizontalSum32(_rowX));
                y = _mm256_add_epi64(y, HorizontalSum32(_rowY));
                xx = _mm256_add_epi64(xx, HorizontalSum32(_rowXX));
                xy = _mm256_add_epi64(xy, HorizontalSum32(_rowXY));
                yy = _mm256_add_epi64(yy, HorizontalSum32(_rowYY));

                mask += stride;
            }
        }

        template <bool align> void GetMomentsLarge(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            __m256i & area, __m256i & x, __m256i & y, __m256i & xx, __m256i & xy, __m256i & yy)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);

            const __m256i K16_I = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            const __m256i _index = _mm256_set1_epi8(index);
            const __m256i tailCol = _mm256_add_epi16(K16_I, _mm256_set1_epi16((uint16_t)(width - A)));

            for(size_t row = 0; row < height; ++row)
            {
                __m256i _col = K16_I;
                __m256i _row = _mm256_set1_epi16((short)row);

                __m256i _rowX = _mm256_setzero_si256();
                __m256i _rowY = _mm256_setzero_si256();
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _mask = _mm256_cmpeq_epi8(Load<align>((__m256i*)(mask + col)), _index);
                    GetMoments8Large(_mask, _row, _col, area, _rowX, _rowY, xx, xy, yy);
                }
                if(alignedWidth != width)
                {
                    __m256i _mask = _mm256_and_si256(_mm256_cmpeq_epi8(Load<false>((__m256i*)(mask + width - A)), _index), tailMask);
                    _col = tailCol;
                    GetMoments8Large(_mask, _row, _col, area, _rowX, _rowY, xx, xy, yy);
                }
                x = _mm256_add_epi64(x, HorizontalSum32(_rowX));
                y = _mm256_add_epi64(y, HorizontalSum32(_rowY));

                mask += stride;
            }
        }

        SIMD_INLINE bool IsSmall(uint64_t width, uint64_t height)
        {
            return 
                width*width*width < 0x300000000ULL && 
                width*width*height < 0x200000000ULL && 
                width*height*height < 0x100000000ULL;
        }

        template <bool align> void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            assert(width >= A && width < SHRT_MAX && height < SHRT_MAX);
            if(align)
                assert(Aligned(mask) && Aligned(stride));

            __m256i _area = _mm256_setzero_si256();
            __m256i _x = _mm256_setzero_si256();
            __m256i _y = _mm256_setzero_si256();
            __m256i _xx = _mm256_setzero_si256();
            __m256i _xy = _mm256_setzero_si256();
            __m256i _yy = _mm256_setzero_si256();

            if(IsSmall(width, height))
                GetMomentsSmall<align>(mask, stride, width, height, index, _area, _x, _y, _xx, _xy, _yy);
            else
                GetMomentsLarge<align>(mask, stride, width, height, index, _area, _x, _y, _xx, _xy, _yy);

            *area = ExtractSum<int64_t>(_area);
            *x = ExtractSum<int64_t>(_x);
            *y = ExtractSum<int64_t>(_y);
            *xx = ExtractSum<int64_t>(_xx);
            *xy = ExtractSum<int64_t>(_xy);
            *yy = ExtractSum<int64_t>(_yy);
        }

        void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            if(Aligned(mask) && Aligned(stride))
                GetMoments<true>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
            else
                GetMoments<false>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        }

        template <bool align> void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);

            memset(sums, 0, sizeof(uint32_t)*height);
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

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
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
                    _p = Allocate(sizeof(uint16_t)*width + sizeof(uint32_t)*width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sums16;
                uint32_t * sums32;
            private:
                void *_p;
            };
        }

        SIMD_INLINE void Sum16(__m256i src8, uint16_t * sums16)
        {
            Store<true>((__m256i*)sums16 + 0, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 0), _mm256_unpacklo_epi8(src8, K_ZERO)));
            Store<true>((__m256i*)sums16 + 1, _mm256_add_epi16(Load<true>((__m256i*)sums16 + 1), _mm256_unpackhi_epi8(src8, K_ZERO)));
        }

        SIMD_INLINE void Sum16To32(const uint16_t * src, uint32_t * dst)
        {
            __m256i lo = LoadPermuted<true>((__m256i*)src + 0);
            __m256i hi = LoadPermuted<true>((__m256i*)src + 1);
            Store<true>((__m256i*)dst + 0, _mm256_add_epi32(Load<true>((__m256i*)dst + 0), _mm256_unpacklo_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 1, _mm256_add_epi32(Load<true>((__m256i*)dst + 1), _mm256_unpacklo_epi16(hi, K_ZERO)));
            Store<true>((__m256i*)dst + 2, _mm256_add_epi32(Load<true>((__m256i*)dst + 2), _mm256_unpackhi_epi16(lo, K_ZERO)));
            Store<true>((__m256i*)dst + 3, _mm256_add_epi32(Load<true>((__m256i*)dst + 3), _mm256_unpackhi_epi16(hi, K_ZERO)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
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
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*alignedLoWidth);
            if(alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint32_t)*(width - alignedLoWidth));
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);

            memset(sums, 0, sizeof(uint32_t)*height);
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
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

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
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
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*alignedLoWidth);
            if(alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint32_t)*(width - alignedLoWidth));
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i src_ = Load<align>((__m256i*)(src + col));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(src_, K_ZERO), fullSum);
                }
                if(width - bodyWidth)
                {
                    const __m256i src_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(src + width - A)));
                    fullSum = _mm256_add_epi64(_mm256_sad_epu8(src_, K_ZERO), fullSum);
                }
                src += stride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        SIMD_INLINE __m256i Square(__m256i src)
        {
            const __m256i lo = _mm256_unpacklo_epi8(src, _mm256_setzero_si256());
            const __m256i hi = _mm256_unpackhi_epi8(src, _mm256_setzero_si256());
            return _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
        }

        template <bool align> void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            for(size_t row = 0; row < height; ++row)
            {
                __m256i rowSum = _mm256_setzero_si256();
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i src_ = Load<align>((__m256i*)(src + col));
                    rowSum = _mm256_add_epi32(rowSum, Square(src_));
                }
                if(width - bodyWidth)
                {
                    const __m256i src_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(src + width - A)));
                    rowSum = _mm256_add_epi32(rowSum, Square(src_));
                }
                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
                src += stride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        SIMD_INLINE __m256i Correlation(__m256i a, __m256i b)
        {
            const __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi8(a, _mm256_setzero_si256()), _mm256_unpacklo_epi8(b, _mm256_setzero_si256()));
            const __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi8(a, _mm256_setzero_si256()), _mm256_unpackhi_epi8(b, _mm256_setzero_si256()));
            return _mm256_add_epi32(lo, hi);
        }

        template <bool align> void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t bodyWidth = AlignLo(width, A);
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();
            for(size_t row = 0; row < height; ++row)
            {
                __m256i rowSum = _mm256_setzero_si256();
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m256i a_ = Load<align>((__m256i*)(a + col));
                    const __m256i b_ = Load<align>((__m256i*)(b + col));
                    rowSum = _mm256_add_epi32(rowSum, Correlation(a_, b_));
                }
                if(width - bodyWidth)
                {
                    const __m256i a_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(a + width - A)));
                    const __m256i b_ = _mm256_and_si256(tailMask, Load<false>((__m256i*)(b + width - A)));
                    rowSum = _mm256_add_epi32(rowSum, Correlation(a_, b_));
                }
                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif// SIMD_AVX2_ENABLE
}