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

#include <math.h>

#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdShiftBilinear.h"
#include "Simd/SimdResizeBilinear.h"

namespace Simd
{
	namespace Base
	{
		namespace
		{
			struct Buffer
			{
				Buffer(size_t width, size_t height)
				{
					_p = Allocate(2*sizeof(int)*(2*width + height));
					ix = (int*)_p;
                    ax = ix + width;
					iy = ax + width;
                    ay = iy + height;
					pbx[0] = (int*)(ay + height);
					pbx[1] = pbx[0] + width;
				}

				~Buffer()
				{
					Free(_p);
				}

				int * ix;
                int * ax;
                int * iy;
                int * ay;
				int * pbx[2];
			private:
				void *_p;
			};
		}

        void EstimateAlphaIndex(size_t srcSize, size_t dstSize, int * indexes, int * alphas, size_t channelCount)
        {
            float scale = (float)srcSize/dstSize;

            for(size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5)*scale - 0.5);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if(index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if(index > (ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                for(size_t c = 0; c < channelCount; c++)
                {
                    size_t offset = i*channelCount + c;
                    indexes[offset] = (int)(channelCount*index + c);
                    alphas[offset] = (int)(alpha * FRACTION_RANGE + 0.5);
                }
            }
        }

		void ResizeBilinear(
			const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
		{
            assert(channelCount >= 1 && channelCount <= 4);

			size_t dstRowSize = channelCount*dstWidth;

			Buffer buffer(dstRowSize, dstHeight);

			EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

			EstimateAlphaIndex(srcWidth, dstWidth, buffer.ix, buffer.ax, channelCount);

			ptrdiff_t previous = -2;

			for(size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
			{
				int fy = buffer.ay[yDst];
				ptrdiff_t sy = buffer.iy[yDst];
				int k = 0;

				if(sy == previous)
					k = 2;
				else if(sy == previous + 1)
				{
					Swap(buffer.pbx[0], buffer.pbx[1]);
					k = 1;
				}

				previous = sy;

				for(; k < 2; k++)
				{
					int* pb = buffer.pbx[k];
					const uchar* ps = src + (sy + k)*srcStride;
					for(size_t x = 0; x < dstRowSize; x++)
					{
						size_t sx = buffer.ix[x];
						int fx = buffer.ax[x];
						int t = ps[sx];
						pb[x] = (t << LINEAR_SHIFT) + (ps[sx + channelCount] - t)*fx;
					}
				}

				if(fy == 0)
					for(size_t xDst = 0; xDst < dstRowSize; xDst++)
						dst[xDst] = ((buffer.pbx[0][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
				else if(fy == FRACTION_RANGE)
					for(size_t xDst = 0; xDst < dstRowSize; xDst++)
						dst[xDst] = ((buffer.pbx[1][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
				else
				{
					for(size_t xDst = 0; xDst < dstRowSize; xDst++)
					{
						int t = buffer.pbx[0][xDst];
						dst[xDst] = ((t << LINEAR_SHIFT) + (buffer.pbx[1][xDst] - t)*fy + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
					}
				}
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
				Buffer(size_t width, size_t height)
				{
					_p = Allocate(sizeof(int)*(2*height + width) + sizeof(short)*4*width);
					ix = (int*)_p;
					ax = (short*)(ix + width);
					pbx[0] = (short*)(ax + 2*width);
					pbx[1] = pbx[0] + width;
                    iy = (int*)(pbx[1] + width);
                    ay = iy + height;
				}

				~Buffer()
				{
					Free(_p);
				}

				int * ix;
				short * ax;
                int * iy;
                int * ay;
				short * pbx[2];
			private:
				void *_p;
			};
		}

		const __m128i K16_FRACTION_ROUND_TERM = SIMD_MM_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

		void EstimateAlphaIndexGrayX(size_t srcSize, size_t dstSize, int * indexes, short * alphas)
		{
			float scale = (float)srcSize/dstSize;

			for(size_t i = 0; i < dstSize; ++i)
			{
				float alpha = (float)((i + 0.5)*scale - 0.5);
				ptrdiff_t index = (ptrdiff_t)::floor(alpha);
				alpha -= index;

				if(index < 0)
				{
					index = 0;
					alpha = 0;
				}

				if(index > (ptrdiff_t)srcSize - 2)
				{
					index = srcSize - 2;
					alpha = 1;
				}

				indexes[i] = (int)index;
				alphas[1] = (short)(alpha * Base::FRACTION_RANGE + 0.5);
				alphas[0] = (short)(Base::FRACTION_RANGE - alphas[1]);
				alphas += 2;
			}
		}

		SIMD_INLINE void InterpolateGrayX(const short* src, const short * alpha, short* dst)
		{
			__m128i s = _mm_load_si128((const __m128i*)src);
			__m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(s, K_ZERO), _mm_load_si128((const __m128i*)alpha + 0));
			__m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(s, K_ZERO), _mm_load_si128((const __m128i*)alpha + 1));
			_mm_store_si128((__m128i*)dst, _mm_packs_epi32(lo, hi));
		}

		SIMD_INLINE void InterpolateGrayY(__m128i s[2], __m128i a[2], uchar *dst)
		{
			__m128i sum = _mm_add_epi16(_mm_mullo_epi16(s[0], a[0]), _mm_mullo_epi16(s[1], a[1]));
			__m128i val = _mm_srli_epi16(_mm_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
			_mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(val, K_ZERO));
		}

		void ResizeBilinearGray(
			const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert(dstWidth >= A);

			size_t bufferWidth = AlignHi(dstWidth, HA);
			size_t alignedWidth = bufferWidth - HA;

			Buffer buffer(bufferWidth, dstHeight);

			Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

			EstimateAlphaIndexGrayX(srcWidth, dstWidth, buffer.ix, buffer.ax);

			ptrdiff_t previous = -2;

			__m128i s[2], a[2];

			for(size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
			{
				a[0] = _mm_set1_epi16(short(Base::FRACTION_RANGE - buffer.ay[yDst]));
				a[1] = _mm_set1_epi16(short(buffer.ay[yDst]));

				ptrdiff_t sy = buffer.iy[yDst];
				int k = 0;

				if(sy == previous)
					k = 2;
				else if(sy == previous + 1)
				{
					Swap(buffer.pbx[0], buffer.pbx[1]);
					k = 1;
				}

				previous = sy;

				for(; k < 2; k++)
				{
					short* pb = buffer.pbx[k];
					const uchar* ps = src + (sy + k)*srcStride;
					for(size_t x = 0; x < dstWidth; x++)
						pb[x] = *(short*)(ps + buffer.ix[x]);

					for(size_t i = 0; i < bufferWidth; i += HA)
					{
						InterpolateGrayX(pb + i, buffer.ax + 2*i, pb + i);
					}
				}

				for(size_t i = 0; i < alignedWidth; i += HA)
				{
					s[0] = _mm_load_si128((__m128i*)(buffer.pbx[0] + i));
					s[1] = _mm_load_si128((__m128i*)(buffer.pbx[1] + i));
					InterpolateGrayY(s, a, dst + i);
				}

				s[0] = _mm_loadu_si128((__m128i*)(buffer.pbx[0] + dstWidth - HA));
				s[1] = _mm_loadu_si128((__m128i*)(buffer.pbx[1] + dstWidth - HA));
				InterpolateGrayY(s, a, dst + dstWidth - HA);
			}
		}

        void ResizeBilinear(
            const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            if(channelCount == 1)
                ResizeBilinearGray(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
        }	
    }
#endif//SIMD_SSE2_ENABLE

	void ResizeBilinear(
		const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
		uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && srcWidth >= Avx2::A)
            Avx2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
        else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && srcWidth >= Sse2::A)
			Sse2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
		else
#endif//SIMD_SSE2_ENABLE
			Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
	}

	void ResizeBilinear(const View & src, View & dst)
	{
		assert(src.format == dst.format);
		assert(src.format == View::Gray8 || src.format == View::Uv16 || src.format == View::Bgr24 || src.format == View::Bgra32);

		if(src.width == dst.width && src.height == dst.height)
		{
			Copy(src, dst);
		}
		else
		{
			ResizeBilinear(src.data, src.width, src.height, src.stride,
				dst.data, dst.width, dst.height, dst.stride, View::SizeOf(src.format));
		}
	}
}

