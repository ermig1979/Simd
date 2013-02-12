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
#include "Simd/SimdInit.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdYuvToBgra.h"

namespace Simd
{
    namespace Base
    {
		SIMD_INLINE void Yuv420ToBgra(const uchar *y, int u, int v, int alpha, uchar * bgra)
		{
			YuvToBgra(y[0], u, v, alpha, bgra);
			YuvToBgra(y[1], u, v, alpha, bgra + 4);
		}

        void RowYuv444ToBgra(uchar *bgra, size_t width, const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            const int *end = y + width;
            for(;y < end; y += 1, u += 1, v += 1, bgra += 4)
            {
                int y0 = y[0] << shift;
                int u0 = u[0] << shift;
                int v0 = v[0] << shift;
                bgra[0] = YuvToBlue(y0, u0);
                bgra[1] = YuvToGreen(y0, u0, v0);
                bgra[2] = YuvToRed(y0, v0);
                bgra[3] = alpha;
            }
        }

        void Yuv444ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            for(size_t row  = 0; row < height; ++row)
            {
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += width;
                v += width;
            }
        }

        void Yuv422ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            assert(height%2 == 0);

            size_t uv_height = height/2;
            for(size_t row  = 0; row < uv_height; ++row)
            {
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                RowYuv444ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += width;
                v += width;
            }
        }

        void RowYuv420ToBgra(uchar *bgra, size_t width, const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            const int *end = y + width;
            for(;y < end; y += 2, u += 1, v += 1, bgra += 8)
            {
                int y0 = y[0] << shift;
                int u0 = u[0] << shift;
                int v0 = v[0] << shift;
                bgra[0] = YuvToBlue(y0, u0);
                bgra[1] = YuvToGreen(y0, u0, v0);
                bgra[2] = YuvToRed(y0, v0);
                bgra[3] = alpha;
                int y1 = y[1] << shift;
                bgra[4] = YuvToBlue(y1, u0);
                bgra[5] = YuvToGreen(y1, u0, v0);
                bgra[6] = YuvToRed(y1, v0);
                bgra[7] = alpha;
            }
        }

        void Yuv420ToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int shift, uchar alpha)
        {
            assert(width%2 == 0 && height%2 == 0);

            size_t uv_width = width/2; 
            size_t uv_height = height/2;
            for(size_t row = 0; row < uv_height; ++row)
            {
                RowYuv420ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                RowYuv420ToBgra(bgra, width, y, u, v, shift, alpha);
                bgra += stride;
                y += width;
                u += uv_width;
                v += uv_width;
            }
        }

        void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
            const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha)
        {
            assert(precision >= 8 && (dx == 1 || dx == 2) && (dy == 1 || dy == 2) && (dy != 1 || dx != 2));

            if(dy == 2)
            {
                if(dx == 2)
                    Yuv420ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
                else
                    Yuv422ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
            }
            else
                Yuv444ToBgra(bgra, width, height, stride, y, u, v, precision - 8, alpha);
        }

		void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			assert((width%2 == 0) && (height%2 == 0) && (width >= 2) && (height >= 2));

			for(size_t row = 0; row < height; row += 2)
			{
				for(size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
				{
					int u_ = u[colUV];
					int v_ = v[colUV];
					Yuv420ToBgra(y + colY, u_, v_, alpha, bgra + colBgra);
					Yuv420ToBgra(y + yStride + colY, u_, v_, alpha, bgra + bgraStride + colBgra);
				}
				y += 2*yStride;
				u += uStride;
				v += vStride;
				bgra += 2*bgraStride;
			}
		}

		void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0, colBgra = 0; col < width; col++, colBgra += 4)
					YuvToBgra(y[col], u[col], v[col], alpha, bgra + colBgra);
				y += yStride;
				u += uStride;
				v += vStride;
				bgra += bgraStride;
			}
		}
   }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
		template <bool align> SIMD_INLINE void AdjustedYuv16ToBgra(__m128i y16, __m128i u16, __m128i v16, 
			const __m128i & a_0, __m128i * bgra)
		{
			const __m128i b16 = AdjustedYuvToBlue16(y16, u16);
			const __m128i g16 = AdjustedYuvToGreen16(y16, u16, v16);
			const __m128i r16 = AdjustedYuvToRed16(y16, v16);
			const __m128i bg8 = _mm_or_si128(b16, _mm_slli_si128(g16, 1));
			const __m128i ra8 = _mm_or_si128(r16, a_0);
			Store<align>(bgra + 0, _mm_unpacklo_epi16(bg8, ra8));
			Store<align>(bgra + 1, _mm_unpackhi_epi16(bg8, ra8));
		}

		template <bool align> SIMD_INLINE void Yuv16ToBgra(__m128i y16, __m128i u16, __m128i v16, 
			const __m128i & a_0, __m128i * bgra)
		{
			AdjustedYuv16ToBgra<align>(AdjustY16(y16), AdjustUV16(u16), AdjustUV16(v16), a_0, bgra);
		}

		template <bool align> SIMD_INLINE void Yuv8ToBgra(__m128i y8, __m128i u8, __m128i v8, const __m128i & a_0, __m128i * bgra)
		{
			Yuv16ToBgra<align>(_mm_unpacklo_epi8(y8, K_ZERO), _mm_unpacklo_epi8(u8, K_ZERO), 
				_mm_unpacklo_epi8(v8, K_ZERO), a_0, bgra + 0);
			Yuv16ToBgra<align>(_mm_unpackhi_epi8(y8, K_ZERO), _mm_unpackhi_epi8(u8, K_ZERO), 
				_mm_unpackhi_epi8(v8, K_ZERO), a_0, bgra + 2);
		}

		template <bool align> SIMD_INLINE void Yuv444ToBgra(const uchar * y, const uchar * u, 
			const uchar * v, const __m128i & a_0, uchar * bgra)
		{
			Yuv8ToBgra<align>(Load<align>((__m128i*)y), Load<align>((__m128i*)u), Load<align>((__m128i*)v), a_0, (__m128i*)bgra);
		}

		template <bool align> void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha)
		{
			assert(width >= A);
			if(align)
			{
				assert(Aligned(y) && Aligned(yStride) && Aligned(u) &&  Aligned(uStride));
				assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
			}

			__m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
			size_t bodyWidth = AlignLo(width, A);
			size_t tail = width - bodyWidth;
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t colYuv = 0, colBgra = 0; colYuv < bodyWidth; colYuv += A, colBgra += QA)
				{
					Yuv444ToBgra<align>(y + colYuv, u + colYuv, v + colYuv, a_0, bgra + colBgra);
				}
				if(tail)
				{
					size_t col = width - A;
					Yuv444ToBgra<false>(y + col, u + col, v + col, a_0, bgra + 4*col);
				}
				y += yStride;
				u += uStride;
				v += vStride;
				bgra += bgraStride;
			}
		}

		template <bool align> SIMD_INLINE void Yuv420ToBgra(const uchar * y, const __m128i & u, const __m128i & v, 
			const __m128i & a_0, uchar * bgra)
		{
			Yuv8ToBgra<align>(Load<align>((__m128i*)y + 0), _mm_unpacklo_epi8(u, u), _mm_unpacklo_epi8(v, v), a_0, (__m128i*)bgra + 0);
			Yuv8ToBgra<align>(Load<align>((__m128i*)y + 1), _mm_unpackhi_epi8(u, u), _mm_unpackhi_epi8(v, v), a_0, (__m128i*)bgra + 4);
		}

		template <bool align> void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha)
		{
			assert((width%2 == 0) && (height%2 == 0) && (width >= DA) && (height >= 2));
			if(align)
			{
				assert(Aligned(y) && Aligned(yStride) && Aligned(u) &&  Aligned(uStride));
				assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
			}

			__m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
			size_t bodyWidth = AlignLo(width, DA);
			size_t tail = width - bodyWidth;
			for(size_t row = 0; row < height; row += 2)
			{
				for(size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
				{
					__m128i u_ = Load<align>((__m128i*)(u + colUV));
					__m128i v_ = Load<align>((__m128i*)(v + colUV));
					Yuv420ToBgra<align>(y + colY, u_, v_, a_0, bgra + colBgra);
					Yuv420ToBgra<align>(y + colY + yStride, u_, v_, a_0, bgra + colBgra + bgraStride);
				}
				if(tail)
				{
					size_t offset = width - DA;
					__m128i u_ = Load<false>((__m128i*)(u + offset/2));
					__m128i v_ = Load<false>((__m128i*)(v + offset/2));
					Yuv420ToBgra<align>(y + offset, u_, v_, a_0, bgra + 4*offset);
					Yuv420ToBgra<align>(y + offset + yStride, u_, v_, a_0, bgra + 4*offset + bgraStride);
				}
				y += 2*yStride;
				u += uStride;
				v += vStride;
				bgra += 2*bgraStride;
			}
		}

		void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			if(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) 
				&& Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
				Yuv420ToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
			else
				Yuv420ToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
		}

		void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
		{
			if(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) 
				&& Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
				Yuv444ToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
			else
				Yuv444ToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
		}
    }
#endif// SIMD_SSE2_ENABLE

	void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
		const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha)
	{
		Base::YuvToBgra(bgra, width, height, stride, y, u, v, dx, dy, precision, alpha);
	}

	void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::DA)
			Sse2::Yuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
		else
#endif//SIMD_SSE2_ENABLE
			Base::Yuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
	}

	void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgra, ptrdiff_t bgraStride, uchar alpha)
	{
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::Yuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
		else
#endif//SIMD_SSE2_ENABLE
			Base::Yuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
	}

	void Yuv444ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha)
	{
		assert(y.width == u.width && y.height == u.height && y.format == u.format);
		assert(y.width == v.width && y.height == v.height && y.format == v.format);
		assert(y.width == bgra.width && y.height == bgra.height);
		assert(y.format == View::Gray8 && bgra.format == View::Bgra32);

		Yuv444ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, 
			y.width, y.height, bgra.data, bgra.stride, alpha);
	}

	void Yuv420ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha)
	{
		assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
		assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
		assert(y.width == bgra.width && y.height == bgra.height);
		assert(y.format == View::Gray8 && bgra.format == View::Bgra32);

		Yuv420ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, 
			y.width, y.height, bgra.data, bgra.stride, alpha);
	}
}