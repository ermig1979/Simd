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
#include "Simd/SimdExtract.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStream.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
		template <bool inversion> __m128i Invert(__m128i value);

		template <> __m128i Invert<true>(__m128i value)
		{
			return _mm_sub_epi8(K_INV_ZERO, value);
		}

		template <> __m128i Invert<false>(__m128i value)
		{
			return value;
		}

		template <bool align, bool stream> void Convert(__m128i src, const __m128 &_1_255, float * dst)
		{
			Sse::Stream<align, stream>(dst + 0, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(src)), _1_255));
			Sse::Stream<align, stream>(dst + 4, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(src)), _1_255));
		}

		template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m128 &_1_255, float * dst)
		{
			__m128i _src = Invert<inversion>(Load<align>((__m128i*)src));
			Convert<align, stream>(UnpackU8<0>(_src), _1_255, dst + 0);
			Convert<align, stream>(UnpackU8<1>(_src), _1_255, dst + 8);
		}

		template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width));

			size_t alignedWidth = AlignLo(width, A);
			__m128 _1_255 = _mm_set1_ps(1.0f / 255.0f);

			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0; col < alignedWidth; col += A)
					Convert<inversion, align, stream>(src + col, _1_255, dst + col);
				if(width != alignedWidth)
					Convert<inversion, false, stream>(src + width - A, _1_255, dst + width - A);
				src += stride;
				dst += width;
			}
            if (stream)
                _mm_mfence();
		}

		template <bool inversion> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
		{
            if (Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width))
            {
                if(width*height*sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, stride, width, height, dst);
                else
                    NeuralConvert<inversion, true, false>(src, stride, width, height, dst);
            }
			else
				NeuralConvert<inversion, false, false>(src, stride, width, height, dst);
		}

		void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
		{
			if (inversion)
				NeuralConvert<true>(src, stride, width, height, dst);
			else
				NeuralConvert<false>(src, stride, width, height, dst);
		}
    }
#endif// SIMD_SSE2_ENABLE
}
