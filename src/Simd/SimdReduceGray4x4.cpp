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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdReduceGray4x4.h"

namespace Simd
{
	namespace Base
	{
		namespace
		{
			struct Buffer
			{
				Buffer(size_t width)
				{
					_p = Allocate(sizeof(int)*2*width);
					src0 = (int*)_p;
					src1 = src0 + width;
				}

				~Buffer()
				{
					Free(_p);
				}

				int * src0;
				int * src1;
			private:
				void *_p;
			};	
		}

		SIMD_INLINE int DivideBy64(int value)
		{
			return (value + 32) >> 6;
		}

		SIMD_INLINE int GaussianBlur(const uchar *src, size_t x0, size_t x1, size_t x2, size_t x3)
		{
			return src[x0] + 3*(src[x1] + src[x2]) + src[x3];
		}

		SIMD_INLINE void ProcessFirstRow(const uchar *src, size_t x0, size_t x1, size_t x2, size_t x3, Buffer & buffer, size_t offset)
		{
			int tmp = GaussianBlur(src, x0, x1, x2, x3);
			buffer.src0[offset] = tmp;
			buffer.src1[offset] = tmp;
		}

		SIMD_INLINE void ProcessMainRow(const uchar *s2, const uchar *s3, size_t x0, size_t x1, size_t x2, size_t x3, Buffer & buffer, uchar* dst, size_t offset)
		{
			int tmp2 = GaussianBlur(s2, x0, x1, x2, x3);
			int tmp3 = GaussianBlur(s3, x0, x1, x2, x3);
			dst[offset] = DivideBy64(buffer.src0[offset] + 3*(buffer.src1[offset] + tmp2) + tmp3);
			buffer.src0[offset] = tmp2;
			buffer.src1[offset] = tmp3;
		}

		void ReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight && srcWidth > 2);

			Buffer buffer(dstWidth);

			ProcessFirstRow(src, 0, 0, 1, 2, buffer, 0);
			size_t srcCol = 2, dstCol = 1;
			for(; srcCol < srcWidth - 2; srcCol += 2, dstCol++)
				ProcessFirstRow(src, srcCol - 1, srcCol, srcCol + 1, srcCol + 2, buffer, dstCol);
			ProcessFirstRow(src, srcCol - 1, srcCol, srcWidth - 1, srcWidth - 1, buffer, dstCol);

			for(size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
			{
				const uchar *src2 = src + srcStride*(row + 1);
				const uchar *src3 = src2 + srcStride;
				if(row >= srcHeight - 2)
				{
					src2 = src + srcStride*(srcHeight - 1);
					src3 = src2;
				}

				ProcessMainRow(src2, src3, 0, 0, 1, 2, buffer, dst, 0);
				size_t srcCol = 2, dstCol = 1;
				for(; srcCol < srcWidth - 2; srcCol += 2, dstCol++)
					ProcessMainRow(src2, src3, srcCol - 1, srcCol, srcCol + 1, srcCol + 2, buffer, dst, dstCol);
				ProcessMainRow(src2, src3, srcCol - 1, srcCol, srcWidth - 1, srcWidth - 1, buffer, dst, dstCol);
			}
		}
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		namespace
		{
			struct Buffer
			{
				Buffer(size_t width)
				{
					_p = Allocate(sizeof(ushort)*4*width);
					src0 = (ushort*)_p;
					src1 = src0 + width;
					src2 = src1 + width;
					src3 = src2 + width;
				}

				~Buffer()
				{
					Free(_p);
				}

				ushort * src0;
				ushort * src1;
				ushort * src2;
				ushort * src3;
			private:
				void * _p;
			};	
		}

		SIMD_INLINE __m128i DivideBy64(__m128i value)
		{
			return _mm_srli_epi16(_mm_add_epi16(value, K16_0020), 6);
		}

		SIMD_INLINE __m128i BinomialSum16(const __m128i & a, const __m128i & b, const __m128i & c, const __m128i & d)
		{
			return _mm_add_epi16(_mm_add_epi16(a, d), _mm_mullo_epi16(_mm_add_epi16(b, c), K16_0003));
		}

		SIMD_INLINE __m128i ReduceColNose(const uchar *src)
		{
			const __m128i t1 = _mm_loadu_si128((__m128i*)src);
			const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
			return BinomialSum16(
				_mm_and_si128(LoadBeforeFirst<1>(t1), K16_00FF),
				_mm_and_si128(t1, K16_00FF),
                _mm_and_si128(t2, K16_00FF),
                _mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
		}

		SIMD_INLINE __m128i ReduceColBody(const uchar *src)
		{
			const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
			const __m128i t2 = _mm_loadu_si128((__m128i*)(src + 1));
			return BinomialSum16(
				_mm_and_si128(t0, K16_00FF),
				_mm_and_si128(_mm_srli_si128(t0, 1), K16_00FF),
				_mm_and_si128(t2, K16_00FF),
				_mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
		}

		template <bool even> SIMD_INLINE __m128i ReduceColTail(const uchar *src);

		template <> SIMD_INLINE __m128i ReduceColTail<true>(const uchar *src)
		{
			const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
			const __m128i t1 = _mm_loadu_si128((__m128i*)src);
			const __m128i t2 = LoadAfterLast<1>(t1);
			return BinomialSum16(
				_mm_and_si128(t0, K16_00FF),
				_mm_and_si128(t1, K16_00FF),
				_mm_and_si128(t2, K16_00FF),
				_mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
		}

		template <> SIMD_INLINE __m128i ReduceColTail<false>(const uchar *src)
		{
			const __m128i t0 = _mm_loadu_si128((__m128i*)(src - 1));
			const __m128i t1 = LoadAfterLast<1>(t0);
			const __m128i t2 = LoadAfterLast<1>(t1);
			return BinomialSum16(
				_mm_and_si128(t0, K16_00FF),
				_mm_and_si128(t1, K16_00FF),
				_mm_and_si128(t2, K16_00FF),
				_mm_and_si128(_mm_srli_si128(t2, 1), K16_00FF));
		}

		template <bool align> SIMD_INLINE __m128i ReduceRow(const Buffer & buffer, size_t offset)
		{
			return _mm_packus_epi16(_mm_and_si128(DivideBy64(BinomialSum16(
				Load<align>((__m128i*)(buffer.src0 + offset)), Load<align>((__m128i*)(buffer.src1 + offset)), 
				Load<align>((__m128i*)(buffer.src2 + offset)), Load<align>((__m128i*)(buffer.src3 + offset)))), K16_00FF), K_ZERO);
		}

		template <bool even> void ReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight && srcWidth > A);

			size_t alignedDstWidth = Simd::AlignLo(dstWidth, A);
			size_t srcTail = Simd::AlignHi(srcWidth - A, 2);

			Buffer buffer(Simd::AlignHi(dstWidth, A));

			__m128i tmp = ReduceColNose(src);
			Store<true>((__m128i*)buffer.src0, tmp);
			Store<true>((__m128i*)buffer.src1, tmp);
			size_t srcCol = A, dstCol = HA;
			for(; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
			{
				tmp = ReduceColBody(src + srcCol);
				Store<true>((__m128i*)(buffer.src0 + dstCol), tmp);
				Store<true>((__m128i*)(buffer.src1 + dstCol), tmp);
			}
			tmp = ReduceColTail<even>(src + srcTail);
			Store<false>((__m128i*)(buffer.src0 + dstWidth - HA), tmp);
			Store<false>((__m128i*)(buffer.src1 + dstWidth - HA), tmp);

			for(size_t row = 0; row < srcHeight; row += 2, dst += dstStride)
			{
				const uchar *src2 = src + srcStride*(row + 1);
				const uchar *src3 = src2 + srcStride;
				if(row >= srcHeight - 2)
				{
					src2 = src + srcStride*(srcHeight - 1);
					src3 = src2;
				}

				Store<true>((__m128i*)buffer.src2, ReduceColNose(src2));
				Store<true>((__m128i*)buffer.src3, ReduceColNose(src3));
				size_t srcCol = A, dstCol = HA;
				for(; srcCol < srcWidth - A; srcCol += A, dstCol += HA)
				{
					Store<true>((__m128i*)(buffer.src2 + dstCol), ReduceColBody(src2 + srcCol));
					Store<true>((__m128i*)(buffer.src3 + dstCol), ReduceColBody(src3 + srcCol));
				}
				Store<false>((__m128i*)(buffer.src2 + dstWidth - HA), ReduceColTail<even>(src2 + srcTail));
				Store<false>((__m128i*)(buffer.src3 + dstWidth - HA), ReduceColTail<even>(src3 + srcTail));

				_mm_storel_epi64((__m128i*)dst, ReduceRow<true>(buffer, 0));

				for(size_t col = HA; col < alignedDstWidth; col += HA)
					_mm_storel_epi64((__m128i*)(dst + col), ReduceRow<true>(buffer, col));

				if(alignedDstWidth != dstWidth)
					_mm_storel_epi64((__m128i*)(dst + dstWidth - HA), ReduceRow<false>(buffer, dstWidth - HA));

				Swap(buffer.src0, buffer.src2);
				Swap(buffer.src1, buffer.src3);
			}
		}

		void ReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			if(Aligned(srcWidth, 2))
				ReduceGray4x4<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
			else
				ReduceGray4x4<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void ReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
		uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && srcWidth > Avx2::DA)
            Avx2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && srcWidth > Sse2::A)
			Sse2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
		else
#endif//SIMD_SSE2_ENABLE
			Base::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
	}
}