/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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

		const int BGR_TO_YUV_AVERAGING_SHIFT = 14;
		const int BGR_TO_YUV_ROUND_TERM = 1 << (BGR_TO_YUV_AVERAGING_SHIFT - 1);
		const int BLUE_TO_Y_WEIGHT = int(0.098*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int GREEN_TO_Y_WEIGHT = int(0.504*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int RED_TO_Y_WEIGHT = int(0.257*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int BLUE_TO_U_WEIGHT = int(0.439*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int GREEN_TO_U_WEIGHT = -int(0.291*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int RED_TO_U_WEIGHT = -int(0.148*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int BLUE_TO_V_WEIGHT = -int(0.071*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int GREEN_TO_V_WEIGHT = -int(0.368*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
		const int RED_TO_V_WEIGHT = int(0.439*(1 << BGR_TO_YUV_AVERAGING_SHIFT) + 0.5);
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

        const __m128i K8_01_FF = SIMD_MM_SET2_EPI8(0x01, 0xFF);

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
        const __m128i K32_0000FFFF = SIMD_MM_SET1_EPI32(0x0000FFFF);
        const __m128i K32_00010000 = SIMD_MM_SET1_EPI32(0x00010000);
        const __m128i K32_01000000 = SIMD_MM_SET1_EPI32(0x01000000);

		const __m128i K64_00000000FFFFFFFF = SIMD_MM_SET2_EPI32(0xFFFFFFFF, 0);

        const __m128i K16_Y_ADJUST = SIMD_MM_SET1_EPI16(Base::Y_ADJUST); 
        const __m128i K16_UV_ADJUST = SIMD_MM_SET1_EPI16(Base::UV_ADJUST);

        const __m128i K16_YRGB_RT = SIMD_MM_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const __m128i K16_VR_0 = SIMD_MM_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const __m128i K16_UG_VG = SIMD_MM_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const __m128i K16_UB_0 = SIMD_MM_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

        const __m128i K16_BY_RY = SIMD_MM_SET2_EPI16(Base::BLUE_TO_Y_WEIGHT, Base::RED_TO_Y_WEIGHT);
        const __m128i K16_GY_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_Y_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const __m128i K16_BU_RU = SIMD_MM_SET2_EPI16(Base::BLUE_TO_U_WEIGHT, Base::RED_TO_U_WEIGHT);
        const __m128i K16_GU_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_U_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const __m128i K16_BV_RV = SIMD_MM_SET2_EPI16(Base::BLUE_TO_V_WEIGHT, Base::RED_TO_V_WEIGHT);
        const __m128i K16_GV_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_V_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        using namespace Sse2;

        const __m128i K8_SHUFFLE_GRAY_TO_BGR0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5);
        const __m128i K8_SHUFFLE_GRAY_TO_BGR1 = SIMD_MM_SETR_EPI8(0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA);
        const __m128i K8_SHUFFLE_GRAY_TO_BGR2 = SIMD_MM_SETR_EPI8(0xA, 0xB, 0xB, 0xB, 0xC, 0xC, 0xC, 0xD, 0xD, 0xD, 0xE, 0xE, 0xE, 0xF, 0xF, 0xF);

        const __m128i K8_SHUFFLE_BLUE_TO_BGR0 = SIMD_MM_SETR_EPI8(0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1, 0x5);
        const __m128i K8_SHUFFLE_BLUE_TO_BGR1 = SIMD_MM_SETR_EPI8( -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1);
        const __m128i K8_SHUFFLE_BLUE_TO_BGR2 = SIMD_MM_SETR_EPI8( -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF,  -1,  -1);

        const __m128i K8_SHUFFLE_GREEN_TO_BGR0 = SIMD_MM_SETR_EPI8( -1, 0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1);
        const __m128i K8_SHUFFLE_GREEN_TO_BGR1 = SIMD_MM_SETR_EPI8(0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA);
        const __m128i K8_SHUFFLE_GREEN_TO_BGR2 = SIMD_MM_SETR_EPI8( -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF,  -1);

        const __m128i K8_SHUFFLE_RED_TO_BGR0 = SIMD_MM_SETR_EPI8( -1,  -1, 0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1);
        const __m128i K8_SHUFFLE_RED_TO_BGR1 = SIMD_MM_SETR_EPI8( -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1);
        const __m128i K8_SHUFFLE_RED_TO_BGR2 = SIMD_MM_SETR_EPI8(0xA,  -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF);

        const __m128i K8_SHUFFLE_BGR0_TO_BLUE = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR1_TO_BLUE = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR2_TO_BLUE = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD);

        const __m128i K8_SHUFFLE_BGR0_TO_GREEN = SIMD_MM_SETR_EPI8(0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR1_TO_GREEN = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR2_TO_GREEN = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE);

        const __m128i K8_SHUFFLE_BGR0_TO_RED = SIMD_MM_SETR_EPI8(0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR1_TO_RED = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m128i K8_SHUFFLE_BGR2_TO_RED = SIMD_MM_SETR_EPI8( -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF);
    }
#endif// SIMD_SSSE3_ENABLE

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        using namespace Ssse3;
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_SSE42_ENABLE    
    namespace Sse42
    {
        using namespace Sse41;
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

        const __m256i K8_01_FF = SIMD_MM256_SET2_EPI8(0x01, 0xFF);

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
        const __m256i K32_00010000 = SIMD_MM256_SET1_EPI32(0x00010000);
        const __m256i K32_01000000 = SIMD_MM256_SET1_EPI32(0x01000000);

        const __m256i K16_Y_ADJUST = SIMD_MM256_SET1_EPI16(Base::Y_ADJUST); 
        const __m256i K16_UV_ADJUST = SIMD_MM256_SET1_EPI16(Base::UV_ADJUST);

        const __m256i K16_YRGB_RT = SIMD_MM256_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const __m256i K16_VR_0 = SIMD_MM256_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const __m256i K16_UG_VG = SIMD_MM256_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const __m256i K16_UB_0 = SIMD_MM256_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

        const __m256i K16_BY_RY = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_Y_WEIGHT, Base::RED_TO_Y_WEIGHT);
        const __m256i K16_GY_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_Y_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const __m256i K16_BU_RU = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_U_WEIGHT, Base::RED_TO_U_WEIGHT);
        const __m256i K16_GU_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_U_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const __m256i K16_BV_RV = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_V_WEIGHT, Base::RED_TO_V_WEIGHT);
        const __m256i K16_GV_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_V_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);

        const __m256i K8_SHUFFLE_GRAY_TO_BGR0 = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5,
            0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA);
        const __m256i K8_SHUFFLE_GRAY_TO_BGR1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7,
            0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xA, 0xB, 0xB, 0xB, 0xC, 0xC, 0xC, 0xD);
        const __m256i K8_SHUFFLE_GRAY_TO_BGR2 = SIMD_MM256_SETR_EPI8(
            0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA,
            0xA, 0xB, 0xB, 0xB, 0xC, 0xC, 0xC, 0xD, 0xD, 0xD, 0xE, 0xE, 0xE, 0xF, 0xF, 0xF);

        const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR0 = SIMD_MM256_SETR_EPI8(
            0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1, 0x5,
             -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1);
        const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR1 = SIMD_MM256_SETR_EPI8(
              -1, 0x3,  -1,  -1, 0x4,  -1,  -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1,
             0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD);
        const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR2 = SIMD_MM256_SETR_EPI8(
            -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1,
            -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF,  -1,  -1);

        const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR0 = SIMD_MM256_SETR_EPI8(
            -1, 0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1,
            0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA);
        const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR1 = SIMD_MM256_SETR_EPI8(
            -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,
            -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1);
        const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR2 = SIMD_MM256_SETR_EPI8(
            0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,
             -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF,  -1);

        const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR0 = SIMD_MM256_SETR_EPI8(
            -1,  -1, 0x0,  -1,  -1, 0x1,  -1,  -1, 0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,
            -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1);
        const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR1 = SIMD_MM256_SETR_EPI8(
            0x2,  -1,  -1, 0x3,  -1,  -1, 0x4,  -1,  -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,
             -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1, 0xA,  -1,  -1, 0xB,  -1,  -1, 0xC,  -1);
        const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR2 = SIMD_MM256_SETR_EPI8(
             -1, 0x5,  -1,  -1, 0x6,  -1,  -1, 0x7,  -1,  -1, 0x8,  -1,  -1, 0x9,  -1,  -1,
            0xA,  -1,  -1, 0xB,  -1,  -1, 0xC,  -1,  -1, 0xD,  -1,  -1, 0xE,  -1,  -1, 0xF);

        const __m256i K8_SHUFFLE_BGR0_TO_BLUE = SIMD_MM256_SETR_EPI8(
            0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR1_TO_BLUE = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD,
            0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR2_TO_BLUE = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD);

        const __m256i K8_SHUFFLE_BGR0_TO_GREEN = SIMD_MM256_SETR_EPI8(
            0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR1_TO_GREEN = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE,
            0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR2_TO_GREEN = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x2, 0x5, 0x8, 0xB, 0xE);

        const __m256i K8_SHUFFLE_BGR0_TO_RED = SIMD_MM256_SETR_EPI8(
            0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR1_TO_RED = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF,
            0x2, 0x5, 0x8, 0xB, 0xE,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1);
        const __m256i K8_SHUFFLE_BGR2_TO_RED = SIMD_MM256_SETR_EPI8(
            -1,  -1,  -1,  -1,  -1, 0x1, 0x4, 0x7, 0xA, 0xD,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF);
	}
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_VMX_ENABLE    
    namespace Vmx
    {
        typedef __vector int8_t v128_s8;
        typedef __vector uint8_t v128_u8;
        typedef __vector int16_t v128_s16;
        typedef __vector uint16_t v128_u16;
        typedef __vector int32_t v128_s32;
        typedef __vector uint32_t v128_u32;
        typedef __vector float v128_f32;

        const size_t A = sizeof(v128_u8);
        const size_t DA = 2*A;
        const size_t QA = 4*A;
        const size_t OA = 8*A;
        const size_t HA = A/2;

        const v128_u8 K8_00 = SIMD_VEC_SET1_EPI8(0x00);
        const v128_u8 K8_01 = SIMD_VEC_SET1_EPI8(0x01);
        const v128_u8 K8_02 = SIMD_VEC_SET1_EPI8(0x02);
        const v128_u8 K8_04 = SIMD_VEC_SET1_EPI8(0x04);
        const v128_u8 K8_08 = SIMD_VEC_SET1_EPI8(0x08);
        const v128_u8 K8_10 = SIMD_VEC_SET1_EPI8(0x10);
        const v128_u8 K8_20 = SIMD_VEC_SET1_EPI8(0x20);
        const v128_u8 K8_40 = SIMD_VEC_SET1_EPI8(0x40);
        const v128_u8 K8_80 = SIMD_VEC_SET1_EPI8(0x80);
        const v128_u8 K8_FF = SIMD_VEC_SET1_EPI8(0xFF);

        const v128_u16 K16_0000 = SIMD_VEC_SET1_EPI16(0x0000);
        const v128_u16 K16_0001 = SIMD_VEC_SET1_EPI16(0x0001);
        const v128_u16 K16_0002 = SIMD_VEC_SET1_EPI16(0x0002);
        const v128_u16 K16_0003 = SIMD_VEC_SET1_EPI16(0x0003);
        const v128_u16 K16_0004 = SIMD_VEC_SET1_EPI16(0x0004);
        const v128_u16 K16_0005 = SIMD_VEC_SET1_EPI16(0x0005);
        const v128_u16 K16_0006 = SIMD_VEC_SET1_EPI16(0x0006);
        const v128_u16 K16_0008 = SIMD_VEC_SET1_EPI16(0x0008);
        const v128_u16 K16_0010 = SIMD_VEC_SET1_EPI16(0x0010);
        const v128_u16 K16_0020 = SIMD_VEC_SET1_EPI16(0x0020);
        const v128_u16 K16_0080 = SIMD_VEC_SET1_EPI16(0x0080);
        const v128_u16 K16_00FF = SIMD_VEC_SET1_EPI16(0x00FF);
        const v128_u16 K16_FFFF = SIMD_VEC_SET1_EPI16(0xFFFF);

        const v128_u32 K32_00000000 = SIMD_VEC_SET1_EPI32(0x00000000);

        const v128_f32 K_0_0f = SIMD_VEC_SET1_PS(0.0f);

        const v128_s16 K16_Y_ADJUST = SIMD_VEC_SET1_EPI16(Base::Y_ADJUST); 
        const v128_s16 K16_UV_ADJUST = SIMD_VEC_SET1_EPI16(Base::UV_ADJUST);

        const v128_s16 K16_YRGB_RT = SIMD_VEC_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
        const v128_s16 K16_VR_0 = SIMD_VEC_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
        const v128_s16 K16_UG_VG = SIMD_VEC_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
        const v128_s16 K16_UB_0 = SIMD_VEC_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

        const v128_u32 K32_YUV_TO_BGR_AVERAGING_SHIFT = SIMD_VEC_SET1_EPI32(Base::YUV_TO_BGR_AVERAGING_SHIFT);

        const v128_s16 K16_BY_RY = SIMD_VEC_SET2_EPI16(Base::BLUE_TO_Y_WEIGHT, Base::RED_TO_Y_WEIGHT);
        const v128_s16 K16_GY_RT = SIMD_VEC_SET2_EPI16(Base::GREEN_TO_Y_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const v128_s16 K16_BU_RU = SIMD_VEC_SET2_EPI16(Base::BLUE_TO_U_WEIGHT, Base::RED_TO_U_WEIGHT);
        const v128_s16 K16_GU_RT = SIMD_VEC_SET2_EPI16(Base::GREEN_TO_U_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
        const v128_s16 K16_BV_RV = SIMD_VEC_SET2_EPI16(Base::BLUE_TO_V_WEIGHT, Base::RED_TO_V_WEIGHT);
        const v128_s16 K16_GV_RT = SIMD_VEC_SET2_EPI16(Base::GREEN_TO_V_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);

        const v128_u32 K32_BGR_TO_YUV_AVERAGING_SHIFT = SIMD_VEC_SET1_EPI32(Base::BGR_TO_YUV_AVERAGING_SHIFT);

        //(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
        const v128_u8 K8_PERM_LOAD_BEFORE_FIRST_1 = SIMD_VEC_SETR_EPI8(0x0, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE);
        const v128_u8 K8_PERM_LOAD_BEFORE_FIRST_2 = SIMD_VEC_SETR_EPI8(0x0, 0x1, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD);
        const v128_u8 K8_PERM_LOAD_BEFORE_FIRST_3 = SIMD_VEC_SETR_EPI8(0x0, 0x1, 0x2, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC);
        const v128_u8 K8_PERM_LOAD_BEFORE_FIRST_4 = SIMD_VEC_SETR_EPI8(0x0, 0x1, 0x2, 0x3, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB);

        const v128_u8 K8_PERM_LOAD_AFTER_LAST_1 = SIMD_VEC_SETR_EPI8(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0xF);
        const v128_u8 K8_PERM_LOAD_AFTER_LAST_2 = SIMD_VEC_SETR_EPI8(0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0xE, 0xF);
        const v128_u8 K8_PERM_LOAD_AFTER_LAST_3 = SIMD_VEC_SETR_EPI8(0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0xD, 0xE, 0xF);
        const v128_u8 K8_PERM_LOAD_AFTER_LAST_4 = SIMD_VEC_SETR_EPI8(0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0xC, 0xD, 0xE, 0xF);

        const v128_u8 K8_PERM_UNPACK_LO_U8 = SIMD_VEC_SETR_EPI8(0x10, 0x00, 0x11, 0x01, 0x12, 0x02, 0x13, 0x03, 0x14, 0x04, 0x15, 0x05, 0x16, 0x06, 0x17, 0x07);
        const v128_u8 K8_PERM_UNPACK_HI_U8 = SIMD_VEC_SETR_EPI8(0x18, 0x08, 0x19, 0x09, 0x1A, 0x0A, 0x1B, 0x0B, 0x1C, 0x0C, 0x1D, 0x0D, 0x1E, 0x0E, 0x1F, 0x0F);

        const v128_u8 K8_PERM_UNPACK_LO_U16 = SIMD_VEC_SETR_EPI8(0x10, 0x11, 0x00, 0x01, 0x12, 0x13, 0x02, 0x03, 0x14, 0x15, 0x04, 0x05, 0x16, 0x17, 0x06, 0x07);
        const v128_u8 K8_PERM_UNPACK_HI_U16 = SIMD_VEC_SETR_EPI8(0x18, 0x19, 0x08, 0x09, 0x1A, 0x1B, 0x0A, 0x0B, 0x1C, 0x1D, 0x0C, 0x0D, 0x1E, 0x1F, 0x0E, 0x0F);

        const v128_u8 K8_PERM_MUL_HI_U16 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x10, 0x11, 0x04, 0x05, 0x14, 0x15, 0x08, 0x09, 0x18, 0x19, 0x0C, 0x0D, 0x1C, 0x1D);

        const v128_u8 K8_PERM_INTERLEAVE_BGR_00 = SIMD_VEC_SETR_EPI8(0x00, 0x10, 0x10, 0x01, 0x11, 0x11, 0x02, 0x12, 0x12, 0x03, 0x13, 0x13, 0x04, 0x14, 0x14, 0x05);
        const v128_u8 K8_PERM_INTERLEAVE_BGR_01 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x10, 0x03, 0x04, 0x11, 0x06, 0x07, 0x12, 0x09, 0x0A, 0x13, 0x0C, 0x0D, 0x14, 0x0F);
        const v128_u8 K8_PERM_INTERLEAVE_BGR_10 = SIMD_VEC_SETR_EPI8(0x15, 0x15, 0x06, 0x16, 0x16, 0x07, 0x17, 0x17, 0x08, 0x18, 0x18, 0x09, 0x19, 0x19, 0x0A, 0x1A);
        const v128_u8 K8_PERM_INTERLEAVE_BGR_11 = SIMD_VEC_SETR_EPI8(0x00, 0x15, 0x02, 0x03, 0x16, 0x05, 0x06, 0x17, 0x08, 0x09, 0x18, 0x0B, 0x0C, 0x19, 0x0E, 0x0F);
        const v128_u8 K8_PERM_INTERLEAVE_BGR_20 = SIMD_VEC_SETR_EPI8(0x1A, 0x0B, 0x1B, 0x1B, 0x0C, 0x1C, 0x1C, 0x0D, 0x1D, 0x1D, 0x0E, 0x1E, 0x1E, 0x0F, 0x1F, 0x1F);
        const v128_u8 K8_PERM_INTERLEAVE_BGR_21 = SIMD_VEC_SETR_EPI8(0x1A, 0x01, 0x02, 0x1B, 0x04, 0x05, 0x1C, 0x07, 0x08, 0x1D, 0x0A, 0x0B, 0x1E, 0x0D, 0x0E, 0x1F);

        const v128_u8 K8_PERM_BGR_TO_BLUE_0 = SIMD_VEC_SETR_EPI8(0x00, 0x03, 0x06, 0x09, 0x0C, 0x0F, 0x12, 0x15, 0x18, 0x1B, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_BGR_TO_BLUE_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x11, 0x14, 0x17, 0x1A, 0x1D);
        const v128_u8 K8_PERM_BGR_TO_GREEN_0 = SIMD_VEC_SETR_EPI8(0x01, 0x04, 0x07, 0x0A, 0x0D, 0x10, 0x13, 0x16, 0x19, 0x1C, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_BGR_TO_GREEN_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x12, 0x15, 0x18, 0x1B, 0x1E);
        const v128_u8 K8_PERM_BGR_TO_RED_0 = SIMD_VEC_SETR_EPI8(0x02, 0x05, 0x08, 0x0B, 0x0E, 0x11, 0x14, 0x17, 0x1A, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_BGR_TO_RED_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x13, 0x16, 0x19, 0x1C, 0x1F);

        const v128_u8 K8_PERM_GRAY_TO_BGR_0 = SIMD_VEC_SETR_EPI8(0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x04, 0x04, 0x04, 0x05);
        const v128_u8 K8_PERM_GRAY_TO_BGR_1 = SIMD_VEC_SETR_EPI8(0x05, 0x05, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0A, 0x0A);
        const v128_u8 K8_PERM_GRAY_TO_BGR_2 = SIMD_VEC_SETR_EPI8(0x0A, 0x0B, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0x0D, 0x0E, 0x0E, 0x0E, 0x0F, 0x0F, 0x0F);
    }
#endif//SIMD_VMX_ENABLE

#ifdef SIMD_VSX_ENABLE    
    namespace Vsx
    {
        using namespace Vmx;
    }
#endif//SIMD_VSX_ENABLE
}
#endif//__SimdConst_h__