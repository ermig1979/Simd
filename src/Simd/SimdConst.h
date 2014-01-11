/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#ifndef __SimdConst_h__
#define __SimdConst_h__

#include "Simd/SimdInit.h"

namespace Simd
{
	const size_t HISTOGRAM_SIZE = UCHAR_MAX + 1;

    namespace Base
    {
        const int LINEAR_SHIFT = 4;
        const int LINEAR_ROUND_TERM = 1 << (LINEAR_SHIFT - 1);

        const int BILINEAR_SHIFT = LINEAR_SHIFT*2;
        const int BILINEAR_ROUND_TERM = 1 << (BILINEAR_SHIFT - 1);

        const int FRACTION_RANGE = 1 << LINEAR_SHIFT;
        const double FRACTION_ROUND_TERM = 0.5/FRACTION_RANGE;

        const float KF_255_DIV_6 = 255.0f/6.0f;

        const int BGR_TO_GRAY_AVERAGING_SHIFT = 14;
        const int BGR_TO_GRAY_ROUND_TERM = 1 << (BGR_TO_GRAY_AVERAGING_SHIFT - 1);
        const int BLUE_TO_GRAY_WEIGHT = int(0.114*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);
        const int GREEN_TO_GRAY_WEIGHT = int(0.587*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);
        const int RED_TO_GRAY_WEIGHT = int(0.299*(1 << BGR_TO_GRAY_AVERAGING_SHIFT) + 0.5);

        const int Y_ADJUST = 16; 
        const int UV_ADJUST = 128;
        const int YUV_TO_BGR_AVERAGING_SHIFT = 13;
        const int YUV_TO_BGR_ROUND_TERM = 1 << (YUV_TO_BGR_AVERAGING_SHIFT - 1); 
        const int Y_TO_RGB_WEIGHT = int(1.164*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int U_TO_BLUE_WEIGHT = int(2.018*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int U_TO_GREEN_WEIGHT = -int(0.391*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int V_TO_GREEN_WEIGHT = -int(0.813*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
        const int V_TO_RED_WEIGHT = int(1.596*(1 << YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        const size_t A = sizeof(__m128i);
        const size_t DA = 2*A;
        const size_t QA = 4*A;
		const size_t OA = 8*A;
        const size_t HA = A/2;

        const __m128i K_ZERO = SIMD_MM_SET1_EPI8(0);
		const __m128i K_INV_ZERO = SIMD_MM_SET1_EPI8(0xFF);

		const __m128i K8_01 = SIMD_MM_SET1_EPI8(0x01);
        const __m128i K8_02 = SIMD_MM_SET1_EPI8(0x02);
        const __m128i K8_04 = SIMD_MM_SET1_EPI8(0x04);
        const __m128i K8_08 = SIMD_MM_SET1_EPI8(0x08);
        const __m128i K8_10 = SIMD_MM_SET1_EPI8(0x10);
        const __m128i K8_20 = SIMD_MM_SET1_EPI8(0x20);
        const __m128i K8_40 = SIMD_MM_SET1_EPI8(0x40);
        const __m128i K8_80 = SIMD_MM_SET1_EPI8(0x80);

		const __m128i K16_0001 = SIMD_MM_SET1_EPI16(0x0001);
		const __m128i K16_0002 = SIMD_MM_SET1_EPI16(0x0002);
		const __m128i K16_0003 = SIMD_MM_SET1_EPI16(0x0003);
		const __m128i K16_0004 = SIMD_MM_SET1_EPI16(0x0004);
		const __m128i K16_0005 = SIMD_MM_SET1_EPI16(0x0005);
        const __m128i K16_0006 = SIMD_MM_SET1_EPI16(0x0006);
		const __m128i K16_0008 = SIMD_MM_SET1_EPI16(0x0008);
		const __m128i K16_0020 = SIMD_MM_SET1_EPI16(0x0020);
		const __m128i K16_0080 = SIMD_MM_SET1_EPI16(0x0080);
        const __m128i K16_00FF = SIMD_MM_SET1_EPI16(0x00FF);
        
		const __m128i K32_000000FF = SIMD_MM_SET1_EPI32(0x000000FF);

		const __m128i K64_00000000FFFFFFFF = SIMD_MM_SET2_EPI32(0xFFFFFFFF, 0);

        const __m128i K16_Y_ADJUST = SIMD_MM_SET1_EPI16(Base::Y_ADJUST); 
        const __m128i K16_UV_ADJUST = SIMD_MM_SET1_EPI16(Base::UV_ADJUST);

        const __m128i K16_YRGB_RT = SIMD_MM_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const __m128i K16_VR_0 = SIMD_MM_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const __m128i K16_UG_VG = SIMD_MM_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const __m128i K16_UB_0 = SIMD_MM_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        using namespace Sse2;
    }
#endif// SIMD_SSSE3_ENABLE

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        using namespace Sse2;
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_SSE42_ENABLE    
    namespace Sse42
    {
        using namespace Sse2;
    }
#endif// SIMD_SSE42_ENABLE

#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
		const size_t A = sizeof(__m256i);
		const size_t DA = 2*A;
		const size_t QA = 4*A;
		const size_t OA = 8*A;
		const size_t HA = A/2;

		const __m256i K_ZERO = SIMD_MM256_SET1_EPI8(0);
		const __m256i K_INV_ZERO = SIMD_MM256_SET1_EPI8(0xFF);

        const __m256i K8_01 = SIMD_MM256_SET1_EPI8(0x01);
        const __m256i K8_02 = SIMD_MM256_SET1_EPI8(0x02);
        const __m256i K8_04 = SIMD_MM256_SET1_EPI8(0x04);
        const __m256i K8_08 = SIMD_MM256_SET1_EPI8(0x08);
        const __m256i K8_10 = SIMD_MM256_SET1_EPI8(0x10);
        const __m256i K8_20 = SIMD_MM256_SET1_EPI8(0x20);
        const __m256i K8_40 = SIMD_MM256_SET1_EPI8(0x40);
        const __m256i K8_80 = SIMD_MM256_SET1_EPI8(0x80);

		const __m256i K16_0001 = SIMD_MM256_SET1_EPI16(0x0001);
        const __m256i K16_0002 = SIMD_MM256_SET1_EPI16(0x0002);
        const __m256i K16_0003 = SIMD_MM256_SET1_EPI16(0x0003);
        const __m256i K16_0004 = SIMD_MM256_SET1_EPI16(0x0004);
        const __m256i K16_0005 = SIMD_MM256_SET1_EPI16(0x0005);
        const __m256i K16_0006 = SIMD_MM256_SET1_EPI16(0x0006);
        const __m256i K16_0008 = SIMD_MM256_SET1_EPI16(0x0008);
        const __m256i K16_0010 = SIMD_MM256_SET1_EPI16(0x0010);
        const __m256i K16_0018 = SIMD_MM256_SET1_EPI16(0x0018);
        const __m256i K16_0020 = SIMD_MM256_SET1_EPI16(0x0020);
        const __m256i K16_0080 = SIMD_MM256_SET1_EPI16(0x0080);
        const __m256i K16_00FF = SIMD_MM256_SET1_EPI16(0x00FF);

		const __m256i K32_000000FF = SIMD_MM256_SET1_EPI32(0x000000FF);
        const __m256i K32_0000FFFF = SIMD_MM256_SET1_EPI32(0x0000FFFF);

        const __m256i K16_Y_ADJUST = SIMD_MM256_SET1_EPI16(Base::Y_ADJUST); 
        const __m256i K16_UV_ADJUST = SIMD_MM256_SET1_EPI16(Base::UV_ADJUST);

        const __m256i K16_YRGB_RT = SIMD_MM256_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const __m256i K16_VR_0 = SIMD_MM256_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const __m256i K16_UG_VG = SIMD_MM256_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const __m256i K16_UB_0 = SIMD_MM256_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);
	}
#endif// SIMD_AVX2_ENABLE
}
#endif//__SimdConst_h__