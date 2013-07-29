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
#ifndef __SimdYuvToBgr_h__
#define __SimdYuvToBgr_h__

#include "Simd/SimdView.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        const int Y_ADJUST = 16; 
        const int UV_ADJUST = 128;
        const int YUV_TO_BGR_AVERAGING_SHIFT = 13;
        const int YUV_TO_BGR_ROUND_TERM = 1 << (YUV_TO_BGR_AVERAGING_SHIFT - 1); 
        const int Y_TO_RGB_WEIGHT = int(1.164*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int U_TO_BLUE_WEIGHT = int(2.018*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int U_TO_GREEN_WEIGHT = -int(0.391*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int V_TO_GREEN_WEIGHT = -int(0.813*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int V_TO_RED_WEIGHT = int(1.596*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);

        SIMD_INLINE int YuvToBlue(int y, int u)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + U_TO_BLUE_WEIGHT*(u - UV_ADJUST) + 
                YUV_TO_BGR_ROUND_TERM) >> YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int YuvToGreen(int y, int u, int v)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + U_TO_GREEN_WEIGHT*(u - UV_ADJUST) + 
                V_TO_GREEN_WEIGHT*(v - UV_ADJUST) + YUV_TO_BGR_ROUND_TERM) >> YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int YuvToRed(int y, int v)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + V_TO_RED_WEIGHT*(v - UV_ADJUST) + 
                YUV_TO_BGR_ROUND_TERM) >> YUV_TO_BGR_AVERAGING_SHIFT);
        }

		SIMD_INLINE void YuvToBgr(int y, int u, int v, uchar * bgr)
		{
			bgr[0] = YuvToBlue(y, u);
			bgr[1] = YuvToGreen(y, u, v);
			bgr[2] = YuvToRed(y, v);
		}

		void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride);

		SIMD_INLINE void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride, bool flip)
		{
			if(flip)
			{
				bgr = bgr + bgrStride*(height - 2);
				bgrStride = -bgrStride;
			}
			Yuv420ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
		}

		void Yuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
			size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride);
    }

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		const __m128i K16_Y_ADJUST = SIMD_MM_SET1_EPI16(Base::Y_ADJUST); 
		const __m128i K16_UV_ADJUST = SIMD_MM_SET1_EPI16(Base::UV_ADJUST);

		const __m128i K16_YRGB_RT = SIMD_MM_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
		const __m128i K16_VR_0 = SIMD_MM_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
		const __m128i K16_UG_VG = SIMD_MM_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
		const __m128i K16_UB_0 = SIMD_MM_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

		SIMD_INLINE __m128i AdjustY16(__m128i y16)
		{
			return _mm_subs_epi16(y16, K16_Y_ADJUST);
		}

		SIMD_INLINE __m128i AdjustUV16(__m128i uv16)
		{
			return _mm_subs_epi16(uv16, K16_UV_ADJUST);
		}

		SIMD_INLINE __m128i AdjustedYuvToRed32(__m128i y16_1, __m128i v16_0)
		{
			return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, K16_YRGB_RT), 
				_mm_madd_epi16(v16_0, K16_VR_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
		}

		SIMD_INLINE __m128i AdjustedYuvToRed16(__m128i y16, __m128i v16)
		{
			return SaturateI16ToU8(_mm_packs_epi32(
				AdjustedYuvToRed32(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(v16, K_ZERO)), 
				AdjustedYuvToRed32(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(v16, K_ZERO))));
		}

		SIMD_INLINE __m128i AdjustedYuvToGreen32(__m128i y16_1, __m128i u16_v16)
		{
			return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, K16_YRGB_RT), 
				_mm_madd_epi16(u16_v16, K16_UG_VG)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
		}

		SIMD_INLINE __m128i AdjustedYuvToGreen16(__m128i y16, __m128i u16, __m128i v16)
		{
			return SaturateI16ToU8(_mm_packs_epi32(
				AdjustedYuvToGreen32(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(u16, v16)), 
				AdjustedYuvToGreen32(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(u16, v16))));
		}

		SIMD_INLINE __m128i AdjustedYuvToBlue32(__m128i y16_1, __m128i u16_0)
		{
			return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, K16_YRGB_RT), 
				_mm_madd_epi16(u16_0, K16_UB_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
		}

		SIMD_INLINE __m128i AdjustedYuvToBlue16(__m128i y16, __m128i u16)
		{
			return SaturateI16ToU8(_mm_packs_epi32(
				AdjustedYuvToBlue32(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(u16, K_ZERO)), 
				AdjustedYuvToBlue32(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(u16, K_ZERO))));
		}	
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K16_Y_ADJUST = SIMD_MM256_SET1_EPI16(Base::Y_ADJUST); 
        const __m256i K16_UV_ADJUST = SIMD_MM256_SET1_EPI16(Base::UV_ADJUST);

        const __m256i K16_YRGB_RT = SIMD_MM256_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const __m256i K16_VR_0 = SIMD_MM256_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const __m256i K16_UG_VG = SIMD_MM256_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const __m256i K16_UB_0 = SIMD_MM256_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

        SIMD_INLINE __m256i AdjustY16(__m256i y16)
        {
            return _mm256_subs_epi16(y16, K16_Y_ADJUST);
        }

        SIMD_INLINE __m256i AdjustUV16(__m256i uv16)
        {
            return _mm256_subs_epi16(uv16, K16_UV_ADJUST);
        }

        SIMD_INLINE __m256i AdjustedYuvToRed32(__m256i y16_1, __m256i v16_0)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, K16_YRGB_RT), 
                _mm256_madd_epi16(v16_0, K16_VR_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i AdjustedYuvToRed16(__m256i y16, __m256i v16)
        {
            return SaturateI16ToU8(PackI32ToI16(
                AdjustedYuvToRed32(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(v16, K_ZERO)), 
                AdjustedYuvToRed32(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(v16, K_ZERO))));
        }

        SIMD_INLINE __m256i AdjustedYuvToGreen32(__m256i y16_1, __m256i u16_v16)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, K16_YRGB_RT), 
                _mm256_madd_epi16(u16_v16, K16_UG_VG)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i AdjustedYuvToGreen16(__m256i y16, __m256i u16, __m256i v16)
        {
            return SaturateI16ToU8(PackI32ToI16(
                AdjustedYuvToGreen32(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(u16, v16)), 
                AdjustedYuvToGreen32(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(u16, v16))));
        }

        SIMD_INLINE __m256i AdjustedYuvToBlue32(__m256i y16_1, __m256i u16_0)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, K16_YRGB_RT), 
                _mm256_madd_epi16(u16_0, K16_UB_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i AdjustedYuvToBlue16(__m256i y16, __m256i u16)
        {
            return SaturateI16ToU8(PackI32ToI16(
                AdjustedYuvToBlue32(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(u16, K_ZERO)), 
                AdjustedYuvToBlue32(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(u16, K_ZERO))));
        }	
    }
#endif// SIMD_AVX2_ENABLE

	void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride);

	void Yuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
		size_t width, size_t height, uchar * bgr, ptrdiff_t bgrStride);

	void Yuv444ToBgr(const View & y, const View & u, const View & v, View & bgr);

	void Yuv420ToBgr(const View & y, const View & u, const View & v, View & bgr);
}
#endif//__SimdYuvToBgr_h__