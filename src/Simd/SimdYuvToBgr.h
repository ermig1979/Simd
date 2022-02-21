/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

#include "Simd/SimdInit.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        /* Corresponds to BT.601 standard. Uses Kr=0.299, Kb=0.114. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt601
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_A = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.114f * (1.0f - 0.114f) / (1.0f - 0.299f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.299f * (1.0f - 0.299f) / (1.0f - 0.299f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.299f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);

            //-------------------------------------------------------------------------------------

            static const int B_SHIFT = 14;
            static const int B_RANGE = 1 << B_SHIFT;
            static const int B_ROUND = 1 << (B_SHIFT - 1);

            static const int B_2_Y = int(0.114f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_Y = int((1.0f - 0.299f - 0.114f) * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_Y = int(0.299f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_U = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_U = -int(0.5f * (1.0f - 0.299f - 0.114f) / (1 - 0.114f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_U = -int(0.5f * 0.299f / (1 - 0.114f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_V = -int(0.5f * 0.114f / (1 - 0.299f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_V = -int(0.5f * (1.0f - 0.299f - 0.114f) / (1 - 0.299f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_V = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
        };

        /* Corresponds to BT.709 standard. Uses Kr=0.2126, Kb=0.0722. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt709
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_A = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.0722f * (1.0f - 0.0722f) / (1.0f - 0.2126f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.2126f * (1.0f - 0.2126f) / (1.0f - 0.2126f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.2126f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);

            //-------------------------------------------------------------------------------------

            static const int B_SHIFT = 14;
            static const int B_RANGE = 1 << B_SHIFT;
            static const int B_ROUND = 1 << (B_SHIFT - 1);

            static const int B_2_Y = int(0.0722f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_Y = int((1.0f - 0.2126f - 0.0722f) * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_Y = int(0.2126f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_U = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_U = -int(0.5f * (1.0f - 0.2126f - 0.0722f) / (1 - 0.0722f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_U = -int(0.5f * 0.2126f / (1 - 0.0722f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_V = -int(0.5f * 0.0722f / (1 - 0.2126f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_V = -int(0.5f * (1.0f - 0.2126f - 0.0722f) / (1 - 0.2126f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_V = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
        };

        /* Corresponds to BT.2020 standard. Uses Kr=0.2627, Kb=0.0593. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt2020
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_A = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.0593f * (1.0f - 0.0593f) / (1.0f - 0.2627f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.2627f * (1.0f - 0.2126f) / (1.0f - 0.2627f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.2627f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);

            //-------------------------------------------------------------------------------------

            static const int B_SHIFT = 14;
            static const int B_RANGE = 1 << B_SHIFT;
            static const int B_ROUND = 1 << (B_SHIFT - 1);

            static const int B_2_Y = int(0.0593f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_Y = int((1.0f - 0.2627f - 0.0593f) * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_Y = int(0.2627f * (Y_HI - Y_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_U = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_U = -int(0.5f * (1.0f - 0.2627f - 0.0593f) / (1 - 0.0593f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_U = -int(0.5f * 0.2627f / (1 - 0.0593f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int B_2_V = -int(0.5f * 0.0593f / (1 - 0.2627f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int G_2_V = -int(0.5f * (1.0f - 0.2627f - 0.0593f) / (1 - 0.2627f) * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
            static const int R_2_V = int(0.5f * (UV_HI - UV_LO) / 255 * B_RANGE + 0.5f);
        };

        /* Corresponds to T-REC-T.871 standard. Uses Kr=0.299, Kb=0.114. Y, U and V use full range [0..255]. */
        struct Trect871
        {
            static const int Y_LO = 0;
            static const int Y_HI = 255;
            static const int UV_LO = 0;
            static const int UV_HI = 255;
            static const int UV_Z = 128;

            static const int F_SHIFT = 14;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_A = int(1.0f * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.114f) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.114f * (1.0f - 0.114f) / (1.0f - 0.299f - 0.114f) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.299f * (1.0f - 0.299f) / (1.0f - 0.299f - 0.114f) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.299f) * F_RANGE + 0.5f);

            //-------------------------------------------------------------------------------------

            static const int B_SHIFT = 14;
            static const int B_RANGE = 1 << B_SHIFT;
            static const int B_ROUND = 1 << (B_SHIFT - 1);

            static const int B_2_Y = int(0.114f * B_RANGE + 0.5f);
            static const int G_2_Y = int((1.0f - 0.299f - 0.114f) * B_RANGE + 0.5f);
            static const int R_2_Y = int(0.299f * B_RANGE + 0.5f);
            static const int B_2_U = int(0.5f * B_RANGE + 0.5f);
            static const int G_2_U = -int(0.5f * (1.0f - 0.299f - 0.114f) / (1 - 0.114f) * B_RANGE + 0.5f);
            static const int R_2_U = -int(0.5f * 0.299f / (1 - 0.114f) * B_RANGE + 0.5f);
            static const int B_2_V = -int(0.5f * 0.114f / (1 - 0.299f) * B_RANGE + 0.5f);
            static const int G_2_V = -int(0.5f * (1.0f - 0.299f - 0.114f) / (1 - 0.299f) * B_RANGE + 0.5f);
            static const int R_2_V = int(0.5f * B_RANGE + 0.5f);
        };

        //-----------------------------------------------------------------------------------------

        template<class T> SIMD_INLINE int YuvToBlue(int y, int u)
        {
            return RestrictRange((T::Y_2_A * (y - T::Y_LO) + T::U_2_B * (u - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE int YuvToGreen(int y, int u, int v)
        {
            return RestrictRange((T::Y_2_A * (y - T::Y_LO) + T::U_2_G * (u - T::UV_Z) + T::V_2_G * (v - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE int YuvToRed(int y, int v)
        {
            return RestrictRange((T::Y_2_A * (y - T::Y_LO) + T::V_2_R * (v - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE void YuvToBgr(int y, int u, int v, uint8_t* bgr)
        {
            bgr[0] = YuvToBlue<T>(y, u);
            bgr[1] = YuvToGreen<T>(y, u, v);
            bgr[2] = YuvToRed<T>(y, v);
        }

        template<class T> SIMD_INLINE void YuvToBgra(int y, int u, int v, int alpha, uint8_t* bgra)
        {
            bgra[0] = YuvToBlue<T>(y, u);
            bgra[1] = YuvToGreen<T>(y, u, v);
            bgra[2] = YuvToRed<T>(y, v);
            bgra[3] = alpha;
        }

        //-----------------------------------------------------------------------------------------

        template<class T> SIMD_INLINE int BgrToY(int blue, int green, int red)
        {
            return RestrictRange(((T::B_2_Y * blue + T::G_2_Y * green + T::R_2_Y * red + T::B_ROUND) >> T::B_SHIFT) + T::Y_LO);
        }

        template<class T> SIMD_INLINE int BgrToU(int blue, int green, int red)
        {
            return RestrictRange(((T::B_2_U * blue + T::G_2_U * green + T::R_2_U * red + T::B_ROUND) >> T::B_SHIFT) + T::UV_Z);
        }

        template<class T> SIMD_INLINE int BgrToV(int blue, int green, int red)
        {
            return RestrictRange(((T::B_2_V * blue + T::G_2_V * green + T::R_2_V * red + T::B_ROUND) >> T::B_SHIFT) + T::UV_Z);
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template<class T> SIMD_INLINE __m128i YuvToRed32(__m128i y16_1, __m128i v16_0)
        {
            static const __m128i YA_RT = SIMD_MM_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m128i VR_0 = SIMD_MM_SET2_EPI16(T::V_2_R, 0);
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, YA_RT), _mm_madd_epi16(v16_0, VR_0)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m128i YuvToGreen32(__m128i y16_1, __m128i u16_v16)
        {
            static const __m128i YA_RT = SIMD_MM_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m128i UG_VG = SIMD_MM_SET2_EPI16(T::U_2_G, T::V_2_G);
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, YA_RT), _mm_madd_epi16(u16_v16, UG_VG)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m128i YuvToBlue32(__m128i y16_1, __m128i u16_0)
        {
            static const __m128i YA_RT = SIMD_MM_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m128i UB_0 = SIMD_MM_SET2_EPI16(T::U_2_B, 0);
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, YA_RT), _mm_madd_epi16(u16_0, UB_0)), T::F_SHIFT);
        }

        template <class T, int part> SIMD_INLINE __m128i UnpackY(__m128i y)
        {
            static const __m128i Y_LO = SIMD_MM_SET1_EPI16(T::Y_LO);
            return _mm_subs_epi16(UnpackU8<part>(y, K_ZERO), Y_LO);
        }

        template <class T, int part> SIMD_INLINE __m128i UnpackUV(__m128i uv)
        {
            static const __m128i UV_Z = SIMD_MM_SET1_EPI16(T::UV_Z);
            return _mm_subs_epi16(UnpackU8<part>(uv, K_ZERO), UV_Z);
        }

        template <class T> SIMD_INLINE __m128i YuvToRed16(__m128i y16, __m128i v16)
        {
            __m128i lo = YuvToRed32<T>(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(v16, K_ZERO));
            __m128i hi = YuvToRed32<T>(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(v16, K_ZERO));
            return SaturateI16ToU8(_mm_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m128i YuvToGreen16(__m128i y16, __m128i u16, __m128i v16)
        {
            __m128i lo = YuvToGreen32<T>(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(u16, v16));
            __m128i hi = YuvToGreen32<T>(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(u16, v16));
            return SaturateI16ToU8(_mm_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m128i YuvToBlue16(__m128i y16, __m128i u16)
        {
            __m128i lo = YuvToBlue32<T>(_mm_unpacklo_epi16(y16, K16_0001), _mm_unpacklo_epi16(u16, K_ZERO));
            __m128i hi = YuvToBlue32<T>(_mm_unpackhi_epi16(y16, K16_0001), _mm_unpackhi_epi16(u16, K_ZERO));
            return SaturateI16ToU8(_mm_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m128i YuvToRed(__m128i y, __m128i v)
        {
            __m128i lo = YuvToRed16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(v));
            __m128i hi = YuvToRed16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(v));
            return _mm_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m128i YuvToGreen(__m128i y, __m128i u, __m128i v)
        {
            __m128i lo = YuvToGreen16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u), UnpackUV<T, 0>(v));
            __m128i hi = YuvToGreen16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u), UnpackUV<T, 1>(v));
            return _mm_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m128i YuvToBlue(__m128i y, __m128i u)
        {
            __m128i lo = YuvToBlue16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u));
            __m128i hi = YuvToBlue16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u));
            return _mm_packus_epi16(lo, hi);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<class T> SIMD_INLINE __m256i YuvToRed32(__m256i y16_1, __m256i v16_0)
        {
            static const __m256i YA_RT = SIMD_MM256_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m256i VR_0 = SIMD_MM256_SET2_EPI16(T::V_2_R, 0);
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, YA_RT), _mm256_madd_epi16(v16_0, VR_0)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m256i YuvToGreen32(__m256i y16_1, __m256i u16_v16)
        {
            static const __m256i YA_RT = SIMD_MM256_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m256i UG_VG = SIMD_MM256_SET2_EPI16(T::U_2_G, T::V_2_G);
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, YA_RT), _mm256_madd_epi16(u16_v16, UG_VG)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m256i YuvToBlue32(__m256i y16_1, __m256i u16_0)
        {
            static const __m256i YA_RT = SIMD_MM256_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m256i UB_0 = SIMD_MM256_SET2_EPI16(T::U_2_B, 0);
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(y16_1, YA_RT), _mm256_madd_epi16(u16_0, UB_0)), T::F_SHIFT);
        }

        template <class T, int part> SIMD_INLINE __m256i UnpackY(__m256i y)
        {
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);
            return _mm256_subs_epi16(UnpackU8<part>(y, K_ZERO), Y_LO);
        }

        template <class T, int part> SIMD_INLINE __m256i UnpackUV(__m256i uv)
        {
            static const __m256i UV_Z = SIMD_MM256_SET1_EPI16(T::UV_Z);
            return _mm256_subs_epi16(UnpackU8<part>(uv, K_ZERO), UV_Z);
        }

        template <class T> SIMD_INLINE __m256i YuvToRed16(__m256i y16, __m256i v16)
        {
            __m256i lo = YuvToRed32<T>(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(v16, K_ZERO));
            __m256i hi = YuvToRed32<T>(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(v16, K_ZERO));
            return SaturateI16ToU8(_mm256_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m256i YuvToGreen16(__m256i y16, __m256i u16, __m256i v16)
        {
            __m256i lo = YuvToGreen32<T>(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(u16, v16));
            __m256i hi = YuvToGreen32<T>(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(u16, v16));
            return SaturateI16ToU8(_mm256_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m256i YuvToBlue16(__m256i y16, __m256i u16)
        {
            __m256i lo = YuvToBlue32<T>(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(u16, K_ZERO));
            __m256i hi = YuvToBlue32<T>(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(u16, K_ZERO));
            return SaturateI16ToU8(_mm256_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m256i YuvToRed(__m256i y, __m256i v)
        {
            __m256i lo = YuvToRed16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(v));
            __m256i hi = YuvToRed16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(v));
            return _mm256_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m256i YuvToGreen(__m256i y, __m256i u, __m256i v)
        {
            __m256i lo = YuvToGreen16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u), UnpackUV<T, 0>(v));
            __m256i hi = YuvToGreen16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u), UnpackUV<T, 1>(v));
            return _mm256_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m256i YuvToBlue(__m256i y, __m256i u)
        {
            __m256i lo = YuvToBlue16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u));
            __m256i hi = YuvToBlue16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u));
            return _mm256_packus_epi16(lo, hi);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<class T> SIMD_INLINE __m512i YuvToRed32(__m512i y16_1, __m512i v16_0)
        {
            static const __m512i YA_RT = SIMD_MM512_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m512i VR_0 = SIMD_MM512_SET2_EPI16(T::V_2_R, 0);
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, YA_RT), _mm512_madd_epi16(v16_0, VR_0)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m512i YuvToGreen32(__m512i y16_1, __m512i u16_v16)
        {
            static const __m512i YA_RT = SIMD_MM512_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m512i UG_VG = SIMD_MM512_SET2_EPI16(T::U_2_G, T::V_2_G);
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, YA_RT), _mm512_madd_epi16(u16_v16, UG_VG)), T::F_SHIFT);
        }

        template<class T> SIMD_INLINE __m512i YuvToBlue32(__m512i y16_1, __m512i u16_0)
        {
            static const __m512i YA_RT = SIMD_MM512_SET2_EPI16(T::Y_2_A, T::F_ROUND);
            static const __m512i UB_0 = SIMD_MM512_SET2_EPI16(T::U_2_B, 0);
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, YA_RT), _mm512_madd_epi16(u16_0, UB_0)), T::F_SHIFT);
        }

        template <class T, int part> SIMD_INLINE __m512i UnpackY(__m512i y)
        {
            static const __m512i Y_LO = SIMD_MM512_SET1_EPI16(T::Y_LO);
            return _mm512_subs_epi16(UnpackU8<part>(y, K_ZERO), Y_LO);
        }

        template <class T, int part> SIMD_INLINE __m512i UnpackUV(__m512i uv)
        {
            static const __m512i UV_Z = SIMD_MM512_SET1_EPI16(T::UV_Z);
            return _mm512_subs_epi16(UnpackU8<part>(uv, K_ZERO), UV_Z);
        }

        template <class T> SIMD_INLINE __m512i YuvToRed16(__m512i y16, __m512i v16)
        {
            __m512i lo = YuvToRed32<T>(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(v16, K_ZERO));
            __m512i hi = YuvToRed32<T>(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(v16, K_ZERO));
            return SaturateI16ToU8(_mm512_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m512i YuvToGreen16(__m512i y16, __m512i u16, __m512i v16)
        {
            __m512i lo = YuvToGreen32<T>(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(u16, v16));
            __m512i hi = YuvToGreen32<T>(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(u16, v16));
            return SaturateI16ToU8(_mm512_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m512i YuvToBlue16(__m512i y16, __m512i u16)
        {
            __m512i lo = YuvToBlue32<T>(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(u16, K_ZERO));
            __m512i hi = YuvToBlue32<T>(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(u16, K_ZERO));
            return SaturateI16ToU8(_mm512_packs_epi32(lo, hi));
        }

        template <class T> SIMD_INLINE __m512i YuvToRed(__m512i y, __m512i v)
        {
            __m512i lo = YuvToRed16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(v));
            __m512i hi = YuvToRed16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(v));
            return _mm512_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m512i YuvToGreen(__m512i y, __m512i u, __m512i v)
        {
            __m512i lo = YuvToGreen16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u), UnpackUV<T, 0>(v));
            __m512i hi = YuvToGreen16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u), UnpackUV<T, 1>(v));
            return _mm512_packus_epi16(lo, hi);
        }

        template <class T> SIMD_INLINE __m512i YuvToBlue(__m512i y, __m512i u)
        {
            __m512i lo = YuvToBlue16<T>(UnpackY<T, 0>(y), UnpackUV<T, 0>(u));
            __m512i hi = YuvToBlue16<T>(UnpackY<T, 1>(y), UnpackUV<T, 1>(u));
            return _mm512_packus_epi16(lo, hi);
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template <class T, int part> SIMD_INLINE int32x4_t YuvToRed(int16x8_t y, int16x8_t v)
        {
            static const int16x4_t YA = SIMD_VEC_SET1_PI16(T::Y_2_A);
            static const int16x4_t VR = SIMD_VEC_SET1_PI16(T::V_2_R);
            static const int32x4_t RT = SIMD_VEC_SET1_PI16(T::F_ROUND);
            return vshrq_n_s32(vmlal_s16(vmlal_s16(RT, Half<part>(y), YA), Half<part>(v), VR), T::F_SHIFT);
        }

        template <class T> SIMD_INLINE int16x8_t YuvToRed(int16x8_t y, int16x8_t v)
        {
            return PackI32(YuvToRed<T, 0>(y, v), YuvToRed<T, 1>(y, v));
        }

        template <class T, int part> SIMD_INLINE int32x4_t YuvToGreen(int16x8_t y, int16x8_t u, int16x8_t v)
        {
            static const int16x4_t YA = SIMD_VEC_SET1_PI16(T::Y_2_A);
            static const int16x4_t UG = SIMD_VEC_SET1_PI16(T::U_2_G);
            static const int16x4_t VG = SIMD_VEC_SET1_PI16(T::V_2_G);
            static const int32x4_t RT = SIMD_VEC_SET1_PI16(T::F_ROUND);
            return vshrq_n_s32(vmlal_s16(vmlal_s16(vmlal_s16(RT, Half<part>(y), YA), Half<part>(u), UG), Half<part>(v), VG), T::F_SHIFT);
        }

        template <class T> SIMD_INLINE int16x8_t YuvToGreen(int16x8_t y, int16x8_t u, int16x8_t v)
        {
            return PackI32(YuvToGreen<T, 0>(y, u, v), YuvToGreen<T, 1>(y, u, v));
        }

        template <class T, int part> SIMD_INLINE int32x4_t YuvToBlue(int16x8_t y, int16x8_t u)
        {
            static const int16x4_t YA = SIMD_VEC_SET1_PI16(T::Y_2_A);
            static const int16x4_t UB = SIMD_VEC_SET1_PI16(T::U_2_B);
            static const int32x4_t RT = SIMD_VEC_SET1_PI16(T::F_ROUND);
            return vshrq_n_s32(vmlal_s16(vmlal_s16(RT, Half<part>(y), YA), Half<part>(u), UB), T::F_SHIFT);
        }

        template <class T> SIMD_INLINE int16x8_t YuvToBlue(int16x8_t y, int16x8_t u)
        {
            return PackI32(YuvToBlue<T, 0>(y, u), YuvToBlue<T, 1>(y, u));
        }

        template <class T, int part> SIMD_INLINE int16x8_t UnpackY(uint8x16_t y)
        {
            static const int16x8_t Y_LO = SIMD_VEC_SET1_EPI16(T::Y_LO);
            return vsubq_s16(vreinterpretq_s16_u16(UnpackU8<part>(y)), Y_LO);
        }

        template <class T, int part> SIMD_INLINE int16x8_t UnpackUV(uint8x16_t uv)
        {
            static const int16x8_t UV_Z = SIMD_VEC_SET1_EPI16(T::UV_Z);
            return vsubq_s16(vreinterpretq_s16_u16(UnpackU8<part>(uv)), UV_Z);
        }

        template <class T> SIMD_INLINE void YuvToBgr(uint8x16_t y, uint8x16_t u, uint8x16_t v, uint8x16x3_t& bgr)
        {
            int16x8_t yLo = UnpackY<T, 0>(y), uLo = UnpackUV<T, 0>(u), vLo = UnpackUV<T, 0>(v);
            int16x8_t yHi = UnpackY<T, 1>(y), uHi = UnpackUV<T, 1>(u), vHi = UnpackUV<T, 1>(v);
            bgr.val[0] = PackSaturatedI16(YuvToBlue<T>(yLo, uLo), YuvToBlue<T>(yHi, uHi));
            bgr.val[1] = PackSaturatedI16(YuvToGreen<T>(yLo, uLo, vLo), YuvToGreen<T>(yHi, uHi, vHi));
            bgr.val[2] = PackSaturatedI16(YuvToRed<T>(yLo, vLo), YuvToRed<T>(yHi, vHi));
        }
    }
#endif
}

#endif//__SimdYuvToBgr_h__
