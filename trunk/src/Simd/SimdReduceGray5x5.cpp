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
#include "Simd/SimdReduceGray5x5.h"

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
					_p = Allocate(sizeof(ushort)*3*width);
					isc0 = (ushort*)_p;
					isc1 = isc0 + width;
					iscp = isc1 + width;
				}

				~Buffer()
				{
					Free(_p);
				}

				ushort * isc0;
				ushort * isc1;
				ushort * iscp;
			private:
				void *_p;
			};	
		}


		/**************************************************************************************************
		*  The Burt & Adelson Reduce operation. This function use 2-D version of algorithm;
		*
		*  Reference:
		*  Frederick M. Waltz and John W.V. Miller. An efficient algorithm for Gaussian blur using 
		*  finite-state machines.
		*  SPIE Conf. on Machine Vision Systems for Inspection and Metrology VII. November 1998.
		*
		*
		*  2-D explanation:
		*
		*  src image pixels:   A  B  C  D  E       dst image pixels:   a     b     c
		*                      F  G  H  I  J
		*                      K  L  M  N  O                           d     e     f
		*                      P  Q  R  S  T
		*                      U  V  W  X  Y                           g     h     i
		*  
		*  Algorithm visits all src image pixels from left to right and top to bottom.
		*  When visiting src pixel Y, the value of e will be written to the dst image.
		*  
		*  State variables before visiting Y:
		*  sr0 = W
		*  sr1 = U + 4V
		*  srp = 4X
		*  sc0[2] = K + 4L + 6M + 4N + O
		*  sc1[2] = (A + 4B + 6C + 4D + E) + 4*(F + 4G + 6H + 4I + J)
		*  scp[2] = 4*(P + 4Q + 6R + 4S + T)
		*  
		*  State variables after visiting Y:
		*  sr0 = Y
		*  sr1 = W + 4X
		*  srp = 4X
		*  sc0[2] = U + 4V + 6W + 4X + Y
		*  sc1[2] = (K + 4L + 6M + 4N + O) + 4*(P + 4Q + 6R + 4S + T)
		*  scp[2] = 4*(P + 4Q + 6R + 4S + T)
		*  e =   1 * (A + 4B + 6C + 4D + E)
		*      + 4 * (F + 4G + 6H + 4I + J)
		*      + 6 * (K + 4L + 6M + 4N + O)
		*      + 4 * (P + 4Q + 6R + 4S + T)
		*      + 1 * (U + 4V + 6W + 4X + Y)
		*  
		*  Updates when visiting (even x, even y) source pixel:
		*  (all updates occur in parallel)
		*  sr0 <= current
		*  sr1 <= sr0 + srp
		*  sc0[x] <= sr1 + 6*sr0 + srp + current
		*  sc1[x] <= sc0[x] + scp[x]
		*  dst(-1,-1) <= sc1[x] + 6*sc0[x] + scp + (new sc0[x])
		*  
		*  Updates when visiting (odd x, even y) source pixel:
		*  srp <= 4*current
		*  
		*  Updates when visiting (even x, odd y) source pixel:
		*  sr0 <= current
		*  sr1 <= sr0 + srp
		*  scp[x] <= 4*(sr1 + 6*sr0 + srp + current)
		*  
		*  Updates when visting (odd x, odd y) source pixel:
		*  srp <= 4*current
		**************************************************************************************************/
		template <bool compensation> SIMD_INLINE int DivideBy256(int value);

		template <> SIMD_INLINE int DivideBy256<true>(int value)
		{
			return (value + 128) >> 8;
		}

		template <> SIMD_INLINE int DivideBy256<false>(int value)
		{
			return value >> 8;
		}

		template <bool compensation> void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight);

			Buffer buffer(dstWidth + 1);

			unsigned short isr0, isr1, isrp;

			const short zeroPixel = 0;

			uchar * dy = dst;
			uchar * dx = dy;
			const uchar * sy = src;
			const uchar * sx = sy;

			bool evenY = true;
			bool evenX = true;
			size_t srcy = 0;
			size_t srcx = 0;
			size_t dstx = 0;

			// First row
			{
				isr0 = *sy;
				isr1 = zeroPixel;
				isrp = (unsigned short)(*sy) * 4;

				// Main pixels in first row
				for (sx = sy, evenX = true, srcx = 0, dstx = 0;  srcx < srcWidth; ++srcx, ++sx)
				{
					unsigned short icurrent(*sx);

					if (evenX)
					{
						buffer.isc0[dstx] = isr1 + 6*isr0 + isrp + icurrent;
						buffer.isc1[dstx] = 5*buffer.isc0[dstx];
						isr1 = isr0 + isrp;
						isr0 = icurrent;
					}
					else
					{
						isrp = icurrent * 4;
						++dstx;
					}
					evenX = !evenX;
				}

				// Last entries in first row
				if (!evenX)
				{
					// previous srcx was even
					++dstx;
					buffer.isc0[dstx] = isr1 + 11*isr0;
					buffer.isc1[dstx] = 5*buffer.isc0[dstx];
				}
				else 
				{
					// previous srcx was odd
					buffer.isc0[dstx] = isr1 + 6*isr0 + isrp + (isrp >> 2);
					buffer.isc1[dstx] = 5*buffer.isc0[dstx];
				}
			}
			sy += srcStride;

			// Main Rows
			{
				for (evenY = false, srcy = 1; srcy < srcHeight; ++srcy, sy += srcStride) 
				{
					isr0 = (unsigned short)(*sy);
					isr1 = zeroPixel;
					isrp = (unsigned short)(*sy) * 4;

					if (evenY)
					{
						// Even-numbered row
						// First entry in row
						sx = sy;
						isr1 = isr0 + isrp;
						isr0 = (unsigned short)(*sx);
						++sx;
						dx = dy;

						register unsigned short * p_isc0 = buffer.isc0;
						register unsigned short * p_isc1 = buffer.isc1;
						register unsigned short * p_iscp = buffer.iscp;

						// Main entries in row
						for (evenX = false, srcx = 1, dstx = 0; srcx < (srcWidth - 1); srcx+=2, ++sx)
						{
							p_isc0++;
							p_isc1++;
							p_iscp++;

							register unsigned short icurrent = (unsigned short)(*sx);

							isrp = icurrent * 4;
							icurrent = (unsigned short)(*(++sx));

							unsigned short ip;
							ip = *p_isc1 + 6*(*p_isc0) + *p_iscp;
							*p_isc1 = *p_isc0 + *p_iscp;
							*p_isc0 = isr1 + 6*isr0 + isrp + icurrent;
							isr1 = isr0 + isrp;
							isr0 = icurrent;
							ip = ip + *p_isc0;
							*dx = DivideBy256<compensation>(ip);
							++dx;
						}		
						dstx += p_isc0 - buffer.isc0;

						//doing the last operation due to even number of operations in previous cycle
						if (!(srcWidth&1))
						{
							register unsigned short icurrent = (unsigned short)(*sx);
							isrp = icurrent* 4;
							++dstx;
							evenX = !evenX;
							++sx;
						}						

						// Last entries in row
						if (!evenX)
						{
							// previous srcx was even
							++dstx;

							unsigned short ip;
							ip = buffer.isc1[dstx] + 6*buffer.isc0[dstx] + buffer.iscp[dstx];
							buffer.isc1[dstx] = buffer.isc0[dstx] + buffer.iscp[dstx];
							buffer.isc0[dstx] = isr1 + 11*isr0;
							ip = ip + buffer.isc0[dstx];
							*dx = DivideBy256<compensation>(ip);
						}
						else
						{
							// Previous srcx was odd
							unsigned short ip;
							ip = buffer.isc1[dstx] + 6*buffer.isc0[dstx] + buffer.iscp[dstx];
							buffer.isc1[dstx] = buffer.isc0[dstx] + buffer.iscp[dstx];
							buffer.isc0[dstx] = isr1 + 6*isr0 + isrp + (isrp >> 2);
							ip = ip + buffer.isc0[dstx];
							*dx = DivideBy256<compensation>(ip);
						}

						dy += dstStride;
					}
					else 
					{
						// First entry in odd-numbered row
						sx = sy;
						isr1 = isr0 + isrp;
						isr0 = (unsigned short)(*sx);
						++sx;

						// Main entries in odd-numbered row
						register unsigned short * p_iscp = buffer.iscp;

						for (evenX = false, srcx = 1, dstx = 0; srcx < (srcWidth - 1); srcx += 2, ++sx)
						{
							register unsigned short icurrent = (unsigned short)(*sx);
							isrp = icurrent * 4;

							p_iscp++;

							icurrent = (unsigned short)(*(++sx));

							*p_iscp = (isr1 + 6*isr0 + isrp + icurrent) * 4;
							isr1 = isr0 + isrp;
							isr0 = icurrent;							
						}
						dstx += p_iscp - buffer.iscp;

						//doing the last operation due to even number of operations in previous cycle
						if (!(srcWidth&1))
						{
							register unsigned short icurrent = (unsigned short)(*sx);
							isrp = icurrent * 4;
							++dstx;
							evenX = !evenX;
							++sx;
						}		

						// Last entries in row
						if (!evenX)
						{
							// previous srcx was even
							++dstx;
							buffer.iscp[dstx] = (isr1 + 11*isr0) * 4;
						}
						else 
						{
							buffer.iscp[dstx] = (isr1 + 6*isr0 + isrp + (isrp >> 2)) * 4;
						}
					}
					evenY = !evenY;
				}
			}

			// Last Rows
			{
				if (!evenY) 
				{
					for (dstx = 1, dx = dy; dstx < (dstWidth + 1); ++dstx, ++dx) 
						*dx = DivideBy256<compensation>(buffer.isc1[dstx] + 11*buffer.isc0[dstx]);
				}
				else
				{
					for (dstx = 1, dx = dy; dstx < (dstWidth + 1); ++dstx, ++dx) 
						*dx = DivideBy256<compensation>(buffer.isc1[dstx] + 6*buffer.isc0[dstx] + buffer.iscp[dstx] + (buffer.iscp[dstx] >> 2));
				}
			}
		}

		void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation)
		{
			if(compensation)
				ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
			else
				ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
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
					_p = Allocate(sizeof(ushort)*(5*width + A));
					in0 = (ushort*)_p;
					in1 = in0 + width;
					out0 = in1 + width;
					out1 = out0 + width;
					dst = out1 + width + HA;
				}

				~Buffer()
				{
					Free(_p);
				}

				ushort * in0;
				ushort * in1;
				ushort * out0;
				ushort * out1;
				ushort * dst;
			private:
				void *_p;
			};	
		}

		template <bool compensation> SIMD_INLINE __m128i DivideBy256(__m128i value);

		template <> SIMD_INLINE __m128i DivideBy256<true>(__m128i value)
		{
			return _mm_srli_epi16(_mm_add_epi16(value, K16_0080), 8);
		}

		template <> SIMD_INLINE __m128i DivideBy256<false>(__m128i value)
		{
			return _mm_srli_epi16(value, 8);
		}

		SIMD_INLINE __m128i LoadUnpacked(const void * src)
		{
			return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)src), K_ZERO);
		}

		template<bool align> SIMD_INLINE void FirstRow5x5(__m128i src, Buffer & buffer, size_t offset)
		{
			Store<align>((__m128i*)(buffer.in0 + offset), src);
			Store<align>((__m128i*)(buffer.in1 + offset), _mm_mullo_epi16(src, K16_0005));
		}

		template<bool align> SIMD_INLINE void FirstRow5x5(const uchar * src, Buffer & buffer, size_t offset)
		{
			FirstRow5x5<align>(LoadUnpacked(src + offset), buffer, offset);
			offset += HA;
			FirstRow5x5<align>(LoadUnpacked(src + offset), buffer, offset);
		}

		template<bool align> SIMD_INLINE void MainRowY5x5(__m128i odd, __m128i even, Buffer & buffer, size_t offset)
		{
			__m128i cp = _mm_mullo_epi16(odd, K16_0004);
			__m128i c0 = Load<align>((__m128i*)(buffer.in0 + offset));
			__m128i c1 = Load<align>((__m128i*)(buffer.in1 + offset));
			Store<align>((__m128i*)(buffer.dst + offset), _mm_add_epi16(even, _mm_add_epi16(c1, _mm_add_epi16(cp, _mm_mullo_epi16(c0, K16_0006)))));
			Store<align>((__m128i*)(buffer.out1 + offset), _mm_add_epi16(c0, cp));
			Store<align>((__m128i*)(buffer.out0 + offset), even);
		}

		template<bool align> SIMD_INLINE void MainRowY5x5(const uchar *odd, const uchar *even, Buffer & buffer, size_t offset)
		{
			MainRowY5x5<align>(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), buffer, offset);
			offset += HA;
			MainRowY5x5<align>(LoadUnpacked(odd + offset), LoadUnpacked(even + offset), buffer, offset);
		}

		template <bool align, bool compensation> SIMD_INLINE __m128i MainRowX5x5(ushort * dst)
		{
			__m128i t0 = _mm_loadu_si128((__m128i*)(dst - 2));
			__m128i t1 = _mm_loadu_si128((__m128i*)(dst - 1));
			__m128i t2 = Load<align>((__m128i*)dst);
			__m128i t3 = _mm_loadu_si128((__m128i*)(dst + 1));
			__m128i t4 = _mm_loadu_si128((__m128i*)(dst + 2));
			t2 = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(t2, K16_0006), _mm_mullo_epi16(_mm_add_epi16(t1, t3), K16_0004)), _mm_add_epi16(t0, t4));
			return DivideBy256<compensation>(t2);
		}

		template <bool align, bool compensation> SIMD_INLINE void MainRowX5x5(Buffer & buffer, size_t offset, uchar *dst)
		{
			__m128i t0 = MainRowX5x5<align, compensation>(buffer.dst + offset);
			__m128i t1 = MainRowX5x5<align, compensation>(buffer.dst + offset + HA);
			t0 = _mm_packus_epi16(_mm_and_si128(_mm_packus_epi16(t0, t1), K16_00FF), K_ZERO); 
			_mm_storel_epi64((__m128i*)dst, t0);
		}

		template <bool compensation> void ReduceGray5x5(
			const uchar* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
			uchar* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight);

			size_t alignedWidth = Simd::AlignLo(srcWidth, A);
			size_t bufferDstTail = Simd::AlignHi(srcWidth - A, 2);

			Buffer buffer(Simd::AlignHi(srcWidth, A));

			for(size_t col = 0; col < alignedWidth; col += A)
				FirstRow5x5<true>(src, buffer, col);
			if(alignedWidth != srcWidth)
				FirstRow5x5<false>(src, buffer, srcWidth - A);
			src += srcStride;

			for(size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2*srcStride)
			{
				const uchar *odd = src - (row < srcHeight ? 0 : srcStride);
				const uchar *even = odd + (row < srcHeight - 1 ? srcStride : 0); 

				for(size_t col = 0; col < alignedWidth; col += A)
					MainRowY5x5<true>(odd, even, buffer, col);
				if(alignedWidth != srcWidth)
					MainRowY5x5<false>(odd, even, buffer, srcWidth - A);

				Swap(buffer.in0, buffer.out0);
				Swap(buffer.in1, buffer.out1);

				buffer.dst[-2] = buffer.dst[0];
				buffer.dst[-1] = buffer.dst[0];
				buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
				buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

				for(size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += A, dstCol += HA)
					MainRowX5x5<true, compensation>(buffer, srcCol, dst + dstCol);
				if(alignedWidth != srcWidth)
					MainRowX5x5<false, compensation>(buffer, bufferDstTail, dst + dstWidth - HA);
			}
		}

		void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation)
		{
			if(compensation)
				ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
			else
				ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
		}
	}
#endif// SIMD_SSE2_ENABLE

	void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
		uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && srcWidth >= Sse2::A)
			Sse2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
		else
#endif//SIMD_SSE2_ENABLE
			Base::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
	}

	void ReduceGray5x5(const View & src, View & dst, bool compensation)
	{
		assert(src.format == View::Gray8 && dst.format == View::Gray8);

		ReduceGray5x5(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation);
	}
}