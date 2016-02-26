/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
             uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;
			uint64x2_t fullSum = K64_0000000000000000;
			uint8x16_t _min = K8_FF;
			uint8x16_t _max = K8_00;
            for(size_t row = 0; row < height; ++row)
            {
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						_min = vminq_u8(_min, _src);
						_max = vmaxq_u8(_max, _src);
						blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
					}
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
				}
                if(width - alignedWidth)
                {
                    const uint8x16_t _src = Load<false>(src + width - A);
					_min = vminq_u8(_min, _src);
					_max = vmaxq_u8(_max, _src);
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(_src, tailMask))));
                }
				fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
				src += stride;
            }

            uint8_t min_buffer[A], max_buffer[A];
            Store<false>(min_buffer, _min);
			Store<false>(max_buffer, _max);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum(fullSum) + width*height/2)/(width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
             uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if(Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

		SIMD_INLINE uint32x4_t MulSum(const uint16x8_t & a, const uint16x8_t & b)
		{
			return vaddq_u32(vmull_u16(Half<0>(a), Half<0>(b)), vmull_u16(Half<1>(a), Half<1>(b)));
		}

		SIMD_INLINE void GetMoments(const uint16x8_t & row, const uint16x8_t & col,
			uint32x4_t & x, uint32x4_t & y, uint32x4_t & xx, uint32x4_t & xy, uint32x4_t & yy)
		{
			x = vaddq_u32(x, vpaddlq_u16(col));
			y = vaddq_u32(y, vpaddlq_u16(row));
			xx = vaddq_u32(xx, MulSum(col, col));
			xy = vaddq_u32(xy, MulSum(col, row));
			yy = vaddq_u32(yy, MulSum(row, row));
		}

		SIMD_INLINE void GetMoments(const uint16x8_t & row, const uint16x8_t & col,
			uint32x4_t & x, uint32x4_t & y, uint64x2_t & xx, uint64x2_t & xy, uint64x2_t & yy)
		{
			x = vaddq_u32(x, vpaddlq_u16(col));
			y = vaddq_u32(y, vpaddlq_u16(row));
			xx = vaddq_u64(xx, vpaddlq_u32(MulSum(col, col)));
			xy = vaddq_u64(xy, vpaddlq_u32(MulSum(col, row)));
			yy = vaddq_u64(yy, vpaddlq_u32(MulSum(row, row)));
		}

		template<class T> SIMD_INLINE void GetMoments(const uint8x16_t & mask, uint16x8_t & row, uint16x8_t & col,
			uint16x8_t & area, uint32x4_t & x, uint32x4_t & y, T & xx, T & xy, T & yy)
		{
			area = vaddq_u16(area, vpaddlq_u8(vandq_u8(K8_01, mask)));

			uint16x8_t lo = vceqq_u16(UnpackU8<0>(mask), K16_00FF);
			GetMoments(vandq_u16(lo, row), vandq_u16(lo, col), x, y, xx, xy, yy);
			col = vaddq_u16(col, K16_0008);

			uint16x8_t hi = vceqq_u16(UnpackU8<1>(mask), K16_00FF);
			GetMoments(vandq_u16(hi, row), vandq_u16(hi, col), x, y, xx, xy, yy);
			col = vaddq_u16(col, K16_0008);
		}

		template <bool align> void GetMomentsSmall(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
			uint32x4_t & area, uint64x2_t & x, uint64x2_t & y, uint64x2_t & xx, uint64x2_t & xy, uint64x2_t & yy)
		{
			size_t alignedWidth = AlignLo(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

			const uint16x8_t K16_I = SIMD_VEC_SETR_EPI16(0, 1, 2, 3, 4, 5, 6, 7);
			const uint8x16_t _index = vdupq_n_u8(index);
			const uint16x8_t tailCol = vaddq_u16(K16_I, vdupq_n_u16((uint16_t)(width - A)));

			for (size_t row = 0; row < height; ++row)
			{
				uint16x8_t _col = K16_I;
				uint16x8_t _row = vdupq_n_u16((uint16_t)row);

				uint16x8_t _rowArea = K16_0000;
				uint32x4_t _rowX = K32_00000000;
				uint32x4_t _rowY = K32_00000000;
				uint32x4_t _rowXX = K32_00000000;
				uint32x4_t _rowXY = K32_00000000;
				uint32x4_t _rowYY = K32_00000000;
				for (size_t col = 0; col < alignedWidth; col += A)
				{
					uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
					GetMoments(_mask, _row, _col, _rowArea, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
				}
				if (alignedWidth != width)
				{
					uint8x16_t _mask = vandq_u8(vceqq_u8(Load<false>(mask + width - A), _index), tailMask);
					_col = tailCol;
					GetMoments(_mask, _row, _col, _rowArea, _rowX, _rowY, _rowXX, _rowXY, _rowYY);
				}
				area = vaddq_u32(area, vpaddlq_u16(_rowArea));
				x = vaddq_u64(x, vpaddlq_u32(_rowX));
				y = vaddq_u64(y, vpaddlq_u32(_rowY));
				xx = vaddq_u64(xx, vpaddlq_u32(_rowXX));
				xy = vaddq_u64(xy, vpaddlq_u32(_rowXY));
				yy = vaddq_u64(yy, vpaddlq_u32(_rowYY));

				mask += stride;
			}
		}

		template <bool align> void GetMomentsLarge(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
			uint32x4_t & area, uint64x2_t & x, uint64x2_t & y, uint64x2_t & xx, uint64x2_t & xy, uint64x2_t & yy)
		{
			size_t alignedWidth = AlignLo(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

			const uint16x8_t K16_I = SIMD_VEC_SETR_EPI16(0, 1, 2, 3, 4, 5, 6, 7);
			const uint8x16_t _index = vdupq_n_u8(index);
			const uint16x8_t tailCol = vaddq_u16(K16_I, vdupq_n_u16((uint16_t)(width - A)));

			for (size_t row = 0; row < height; ++row)
			{
				uint16x8_t _col = K16_I;
				uint16x8_t _row = vdupq_n_u16((uint16_t)row);

				uint16x8_t _rowArea = K16_0000;
				uint32x4_t _rowX = K32_00000000;
				uint32x4_t _rowY = K32_00000000;
				for (size_t col = 0; col < alignedWidth; col += A)
				{
					uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
					GetMoments(_mask, _row, _col, _rowArea, _rowX, _rowY, xx, xy, yy);
				}
				if (alignedWidth != width)
				{
					uint8x16_t _mask = vandq_u8(vceqq_u8(Load<false>(mask + width - A), _index), tailMask);
					_col = tailCol;
					GetMoments(_mask, _row, _col, _rowArea, _rowX, _rowY, xx, xy, yy);
				}
				area = vaddq_u32(area, vpaddlq_u16(_rowArea));
				x = vaddq_u64(x, vpaddlq_u32(_rowX));
				y = vaddq_u64(y, vpaddlq_u32(_rowY));

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
			if (align)
				assert(Aligned(mask) && Aligned(stride));

			uint32x4_t _area = K32_00000000;
			uint64x2_t _x = K64_0000000000000000;
			uint64x2_t _y = K64_0000000000000000;
			uint64x2_t _xx = K64_0000000000000000;
			uint64x2_t _xy = K64_0000000000000000;
			uint64x2_t _yy = K64_0000000000000000;

			if (IsSmall(width, height))
				GetMomentsSmall<align>(mask, stride, width, height, index, _area, _x, _y, _xx, _xy, _yy);
			else
				GetMomentsLarge<align>(mask, stride, width, height, index, _area, _x, _y, _xx, _xy, _yy);

			*area = ExtractSum(_area);
			*x = ExtractSum(_x);
			*y = ExtractSum(_y);
			*xx = ExtractSum(_xx);
			*xy = ExtractSum(_xy);
			*yy = ExtractSum(_yy);
		}

		void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
			uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
		{
			if (Aligned(mask) && Aligned(stride))
				GetMoments<true>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
			else
				GetMoments<false>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
		}

		template <bool align> void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			size_t alignedWidth = AlignLo(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;

			memset(sums, 0, sizeof(uint32_t)*height);
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
					}
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
				}
				if (alignedWidth != width)
				{
					const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(_src)));
				}
				sums[row] = ExtractSum(rowSum);
				src += stride;
			}
		}

		void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			if (Aligned(src) && Aligned(stride))
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

		template <bool align> SIMD_INLINE void Sum16(const uint8x16_t & src, uint16_t * dst)
		{
			Store<align>(dst + 0, vaddq_u16(Load<align>(dst + 0), UnpackU8<0>(src)));
			Store<align>(dst + 8, vaddq_u16(Load<align>(dst + 8), UnpackU8<1>(src)));
		}

		template <bool align> SIMD_INLINE void Sum32(const uint16x8_t & src, uint32_t * dst)
		{
			Store<align>(dst + 0, vaddq_u32(Load<align>(dst + 0), UnpackU16<0>(src)));
			Store<align>(dst + 4, vaddq_u32(Load<align>(dst + 4), UnpackU16<1>(src)));
		}

		template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			size_t alignedLoWidth = AlignLo(width, A);
			size_t alignedHiWidth = AlignHi(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
			size_t stepSize = SCHAR_MAX + 1;
			size_t stepCount = (height + SCHAR_MAX) / stepSize;

			Buffer buffer(alignedHiWidth);
			memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
			for (size_t step = 0; step < stepCount; ++step)
			{
				size_t rowStart = step*stepSize;
				size_t rowEnd = Min(rowStart + stepSize, height);

				memset(buffer.sums16, 0, sizeof(uint16_t)*width);
				for (size_t row = rowStart; row < rowEnd; ++row)
				{
					for (size_t col = 0; col < alignedLoWidth; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						Sum16<true>(_src, buffer.sums16 + col);
					}
					if (alignedLoWidth != width)
					{
						const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
						Sum16<false>(_src, buffer.sums16 + width - A);
					}
					src += stride;
				}

				for (size_t col = 0; col < alignedHiWidth; col += HA)
					Sum32<true>(Load<true>(buffer.sums16 + col), buffer.sums32 + col);
			}
			memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
		}

		void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			if (Aligned(src) && Aligned(stride))
				GetColSums<true>(src, stride, width, height, sums);
			else
				GetColSums<false>(src, stride, width, height, sums);
		}

		template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			size_t alignedWidth = AlignLo(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;

			memset(sums, 0, sizeof(uint32_t)*height);
			const uint8_t * src0 = src;
			const uint8_t * src1 = src + stride;
			height--;
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src0 = Load<align>(src0 + col);
						const uint8x16_t _src1 = Load<align>(src1 + col);
						blockSum = vaddq_u16(blockSum, vpaddlq_u8(vabdq_u8(_src0, _src1)));
					}
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
				}
				if (alignedWidth != width)
				{
					const uint8x16_t _src0 = Load<false>(src0 + width - A);
					const uint8x16_t _src1 = Load<false>(src1 + width - A);
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(vabdq_u8(_src0, _src1), tailMask))));
				}
				sums[row] = ExtractSum(rowSum);
				src0 += stride;
				src1 += stride;
			}
		}

		void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			if (Aligned(src) && Aligned(stride))
				GetAbsDyRowSums<true>(src, stride, width, height, sums);
			else
				GetAbsDyRowSums<false>(src, stride, width, height, sums);
		}

		template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			width--;
			size_t alignedLoWidth = AlignLo(width, A);
			size_t alignedHiWidth = AlignHi(width, A);
			const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
			size_t stepSize = SCHAR_MAX + 1;
			size_t stepCount = (height + SCHAR_MAX) / stepSize;

			Buffer buffer(alignedHiWidth);
			memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
			for (size_t step = 0; step < stepCount; ++step)
			{
				size_t rowStart = step*stepSize;
				size_t rowEnd = Min(rowStart + stepSize, height);

				memset(buffer.sums16, 0, sizeof(uint16_t)*width);
				for (size_t row = rowStart; row < rowEnd; ++row)
				{
					for (size_t col = 0; col < alignedLoWidth; col += A)
					{
						const uint8x16_t _src0 = Load<align>(src + col + 0);
						const uint8x16_t _src1 = Load<false>(src + col + 1);
						Sum16<true>(vabdq_u8(_src0, _src1), buffer.sums16 + col);
					}
					if (alignedLoWidth != width)
					{
						const uint8x16_t _src0 = Load<false>(src + width - A + 0);
						const uint8x16_t _src1 = Load<false>(src + width - A + 1);
						Sum16<false>(vandq_u8(vabdq_u8(_src0, _src1), tailMask), buffer.sums16 + width - A);
					}
					src += stride;
				}

				for (size_t col = 0; col < alignedHiWidth; col += HA)
					Sum32<true>(Load<true>(buffer.sums16 + col), buffer.sums32 + col);
			}
			memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
			sums[width] = 0;
		}

		void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
		{
			if (Aligned(src) && Aligned(stride))
				GetAbsDxColSums<true>(src, stride, width, height, sums);
			else
				GetAbsDxColSums<false>(src, stride, width, height, sums);
		}

		template <bool align> void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
			size_t blockSize = A << 8;
			size_t blockCount = (alignedWidth >> 8) + 1;
			uint64x2_t fullSum = K64_0000000000000000;
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t block = 0; block < blockCount; ++block)
				{
					uint16x8_t blockSum = K16_0000;
					for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
					{
						const uint8x16_t _src = Load<align>(src + col);
						blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
					}
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
				}
				if (width - alignedWidth)
				{
					const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
					rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(_src)));
				}
				fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
				src += stride;
			}
			*sum = ExtractSum(fullSum);
		}

		void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(src) && Aligned(stride))
				ValueSum<true>(src, stride, width, height, sum);
			else
				ValueSum<false>(src, stride, width, height, sum);
		}

		SIMD_INLINE uint16x8_t Square(uint8x8_t value)
		{
			return vmull_u8(value, value);
		}

		SIMD_INLINE uint32x4_t Square(uint8x16_t value)
		{
			uint16x8_t lo = Square(vget_low_u8(value));
			uint16x8_t hi = Square(vget_high_u8(value));
			return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
		}

		template <bool align> void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(src) && Aligned(stride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

			uint64x2_t fullSum = K64_0000000000000000;
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t col = 0; col < alignedWidth; col += A)
					rowSum = vaddq_u32(rowSum, Square(Load<align>(src + col)));
				if (alignedWidth != width)
					rowSum = vaddq_u32(rowSum, Square(vandq_u8(Load<false>(src + width - A), tailMask)));
				fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
				src += stride;
			}
			*sum = ExtractSum(fullSum);
		}

		void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(src) && Aligned(stride))
				SquareSum<true>(src, stride, width, height, sum);
			else
				SquareSum<false>(src, stride, width, height, sum);
		}

		SIMD_INLINE uint32x4_t Correlation(const uint8x16_t & a, const uint8x16_t & b)
		{
			uint16x8_t lo = vmull_u8(Half<0>(a), Half<0>(b));
			uint16x8_t hi = vmull_u8(Half<1>(a), Half<1>(b));
			return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
		}

		template <bool align> void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

			size_t alignedWidth = Simd::AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

			uint64x2_t fullSum = K64_0000000000000000;
			for (size_t row = 0; row < height; ++row)
			{
				uint32x4_t rowSum = K32_00000000;
				for (size_t col = 0; col < alignedWidth; col += A)
				{
					uint8x16_t _a = Load<align>(a + col);
					uint8x16_t _b = Load<align>(b + col);
					rowSum = vaddq_u32(rowSum, Correlation(_a, _b));
				}
				if (alignedWidth != width)
				{
					uint8x16_t _a = vandq_u8(Load<align>(a + width - A), tailMask);
					uint8x16_t _b = vandq_u8(Load<align>(b + width - A), tailMask);
					rowSum = vaddq_u32(rowSum, Correlation(_a, _b));
				}
				fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
				a += aStride;
				b += bStride;
			}
			*sum = ExtractSum(fullSum);
		}

		void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
		{
			if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
				CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
			else
				CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
		}
    }
#endif// SIMD_NEON_ENABLE
}