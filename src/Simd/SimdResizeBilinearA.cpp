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
#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
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

		const __m256i K16_FRACTION_ROUND_TERM = SIMD_MM256_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

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
			__m256i s = LoadPermuted<true>((const __m256i*)src);
			__m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi8(s, K_ZERO), _mm256_load_si256((const __m256i*)alpha + 0));
			__m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi8(s, K_ZERO), _mm256_load_si256((const __m256i*)alpha + 1));
			_mm256_store_si256((__m256i*)dst, PackI32ToI16(lo, hi));
		}

        template <bool align> SIMD_INLINE __m256i InterpolateGrayY16(const Buffer & buffer, size_t offset, __m256i a[2])
        {
            __m256i s0 = Load<align>((__m256i*)(buffer.pbx[0] + offset));
            __m256i s1 = Load<align>((__m256i*)(buffer.pbx[1] + offset));
            __m256i sum = _mm256_add_epi16(_mm256_mullo_epi16(s0, a[0]), _mm256_mullo_epi16(s1, a[1]));
            return _mm256_srli_epi16(_mm256_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

		template <bool align> SIMD_INLINE void InterpolateGrayY(const Buffer & buffer, size_t offset, __m256i a[2], uchar *dst)
		{
            __m256i lo = InterpolateGrayY16<align>(buffer, offset, a);
            __m256i hi = InterpolateGrayY16<align>(buffer, offset + HA, a);
			_mm256_storeu_si256((__m256i*)(dst + offset), PackU16ToU8(lo, hi));
		}

        template <bool align> SIMD_INLINE __m256i GatherGray(const uchar * src, const int * index)
        {
            __m256i lo = _mm256_and_si256(_mm256_i32gather_epi32((int*)src, Load<align>((__m256i*)(index + 0)), 1), K32_0000FFFF);
            __m256i hi = _mm256_and_si256(_mm256_i32gather_epi32((int*)src, Load<align>((__m256i*)(index + 8)), 1), K32_0000FFFF);
            return PackU32ToI16(lo, hi);
        }

		void ResizeBilinearGray(
			const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
			uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
		{
			assert(dstWidth >= A);

			size_t bufferWidth = AlignHi(dstWidth, A);
			size_t alignedWidth = bufferWidth - A;

			Buffer buffer(bufferWidth, dstHeight);

			Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

			EstimateAlphaIndexGrayX(srcWidth, dstWidth, buffer.ix, buffer.ax);

			ptrdiff_t previous = -2;

			__m256i a[2];

			for(size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
			{
				a[0] = _mm256_set1_epi16(short(Base::FRACTION_RANGE - buffer.ay[yDst]));
				a[1] = _mm256_set1_epi16(short(buffer.ay[yDst]));

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

#ifdef SIMD_AVX2_GATHER_DEPRECATE
					for(size_t x = 0; x < dstWidth; x++)
						pb[x] = *(short*)(ps + buffer.ix[x]);
#else//SIMD_AVX2_GATHER_DEPRECATE
                    for(size_t x = 0; x < alignedWidth; x += HA)
                        Store<true>((__m256i*)(pb + x), GatherGray<true>(ps, buffer.ix + x));
                    Store<false>((__m256i*)(pb + dstWidth - A), GatherGray<false>(ps, buffer.ix + dstWidth - A));
                    Store<false>((__m256i*)(pb + dstWidth - HA), GatherGray<false>(ps, buffer.ix + dstWidth - HA));
#endif//SIMD_AVX2_GATHER_DEPRECATE

					for(size_t i = 0; i < bufferWidth; i += HA)
					{
						InterpolateGrayX(pb + i, buffer.ax + 2*i, pb + i);
					}
				}

				for(size_t i = 0; i < alignedWidth; i += A)
					InterpolateGrayY<true>(buffer, i, a, dst);

                InterpolateGrayY<false>(buffer, dstWidth - A, a, dst);
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
#endif//SIMD_AVX2_ENABLE
}

