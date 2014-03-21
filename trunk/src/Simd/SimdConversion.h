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
#ifndef __SimdConversion_h__
#define __SimdConversion_h__

#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int BgrToGray(int blue, int green, int red)
        {
            return (BLUE_TO_GRAY_WEIGHT*blue + GREEN_TO_GRAY_WEIGHT*green + 
                RED_TO_GRAY_WEIGHT*red + BGR_TO_GRAY_ROUND_TERM) >> BGR_TO_GRAY_AVERAGING_SHIFT;
        }

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

        SIMD_INLINE void YuvToBgr(int y, int u, int v, uint8_t * bgr)
        {
            bgr[0] = YuvToBlue(y, u);
            bgr[1] = YuvToGreen(y, u, v);
            bgr[2] = YuvToRed(y, v);
        }

        SIMD_INLINE void YuvToBgra(int y, int u, int v, int alpha, uint8_t * bgra)
        {
            bgra[0] = YuvToBlue(y, u);
            bgra[1] = YuvToGreen(y, u, v);
            bgra[2] = YuvToRed(y, v);
            bgra[3] = alpha;
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
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

        SIMD_INLINE __m128i YuvToRed(__m128i y, __m128i v)
        {
            __m128i lo = AdjustedYuvToRed16(
                AdjustY16(_mm_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpacklo_epi8(v, K_ZERO)));
            __m128i hi = AdjustedYuvToRed16(
                AdjustY16(_mm_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpackhi_epi8(v, K_ZERO)));
            return _mm_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m128i YuvToGreen(__m128i y, __m128i u, __m128i v)
        {
            __m128i lo = AdjustedYuvToGreen16(
                AdjustY16(_mm_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpacklo_epi8(u, K_ZERO)), 
                AdjustUV16(_mm_unpacklo_epi8(v, K_ZERO)));
            __m128i hi = AdjustedYuvToGreen16(
                AdjustY16(_mm_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpackhi_epi8(u, K_ZERO)), 
                AdjustUV16(_mm_unpackhi_epi8(v, K_ZERO)));
            return _mm_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m128i YuvToBlue(__m128i y, __m128i u)
        {
            __m128i lo = AdjustedYuvToBlue16(
                AdjustY16(_mm_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpacklo_epi8(u, K_ZERO)));
            __m128i hi = AdjustedYuvToBlue16(
                AdjustY16(_mm_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm_unpackhi_epi8(u, K_ZERO)));
            return _mm_packus_epi16(lo, hi);
        }

        template <int index> __m128i InterleaveBgr(__m128i blue, __m128i green, __m128i red);

        template<> SIMD_INLINE __m128i InterleaveBgr<0>(__m128i blue, __m128i green, __m128i red)
        {
            return 
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR0), 
                _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR0), 
                _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m128i InterleaveBgr<1>(__m128i blue, __m128i green, __m128i red)
        {
            return 
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR1), 
                _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR1), 
                _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m128i InterleaveBgr<2>(__m128i blue, __m128i green, __m128i red)
        {
            return 
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR2), 
                _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR2), 
                _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR2)));
        }
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
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
            return SaturateI16ToU8(_mm256_packs_epi32(
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
            return SaturateI16ToU8(_mm256_packs_epi32(
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
            return SaturateI16ToU8(_mm256_packs_epi32(
                AdjustedYuvToBlue32(_mm256_unpacklo_epi16(y16, K16_0001), _mm256_unpacklo_epi16(u16, K_ZERO)), 
                AdjustedYuvToBlue32(_mm256_unpackhi_epi16(y16, K16_0001), _mm256_unpackhi_epi16(u16, K_ZERO))));
        }

        SIMD_INLINE __m256i YuvToRed(__m256i y, __m256i v)
        {
            __m256i lo = AdjustedYuvToRed16(
                AdjustY16(_mm256_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpacklo_epi8(v, K_ZERO)));
            __m256i hi = AdjustedYuvToRed16(
                AdjustY16(_mm256_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpackhi_epi8(v, K_ZERO)));
            return _mm256_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m256i YuvToGreen(__m256i y, __m256i u, __m256i v)
        {
            __m256i lo = AdjustedYuvToGreen16(
                AdjustY16(_mm256_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpacklo_epi8(u, K_ZERO)), 
                AdjustUV16(_mm256_unpacklo_epi8(v, K_ZERO)));
            __m256i hi = AdjustedYuvToGreen16(
                AdjustY16(_mm256_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpackhi_epi8(u, K_ZERO)), 
                AdjustUV16(_mm256_unpackhi_epi8(v, K_ZERO)));
            return _mm256_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m256i YuvToBlue(__m256i y, __m256i u)
        {
            __m256i lo = AdjustedYuvToBlue16(
                AdjustY16(_mm256_unpacklo_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpacklo_epi8(u, K_ZERO)));
            __m256i hi = AdjustedYuvToBlue16(
                AdjustY16(_mm256_unpackhi_epi8(y, K_ZERO)), 
                AdjustUV16(_mm256_unpackhi_epi8(u, K_ZERO)));
            return _mm256_packus_epi16(lo, hi);
        }

        template <int index> __m256i InterleaveBgr(__m256i blue, __m256i green, __m256i red);

        template<> SIMD_INLINE __m256i InterleaveBgr<0>(__m256i blue, __m256i green, __m256i red)
        {
            return 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0x44), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR0), 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0x44), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR0), 
                _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0x44), K8_SHUFFLE_PERMUTED_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m256i InterleaveBgr<1>(__m256i blue, __m256i green, __m256i red)
        {
            return 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0x99), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR1), 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0x99), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR1), 
                _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0x99), K8_SHUFFLE_PERMUTED_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m256i InterleaveBgr<2>(__m256i blue, __m256i green, __m256i red)
        {
            return 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0xEE), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR2), 
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0xEE), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR2), 
                _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0xEE), K8_SHUFFLE_PERMUTED_RED_TO_BGR2)));
        }
    }
#endif// SIMD_AVX2_ENABLE
}
#endif//__SimdConversion_h__