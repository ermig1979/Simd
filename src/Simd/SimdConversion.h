/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
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
#ifndef __SimdConversion_h__
#define __SimdConversion_h__

#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int BgrToGray(int blue, int green, int red)
        {
            return (BLUE_TO_GRAY_WEIGHT*blue + GREEN_TO_GRAY_WEIGHT * green +
                RED_TO_GRAY_WEIGHT * red + BGR_TO_GRAY_ROUND_TERM) >> BGR_TO_GRAY_AVERAGING_SHIFT;
        }

        SIMD_INLINE int BgrToY(int blue, int green, int red)
        {
            return RestrictRange(((BLUE_TO_Y_WEIGHT*blue + GREEN_TO_Y_WEIGHT * green + RED_TO_Y_WEIGHT * red +
                BGR_TO_YUV_ROUND_TERM) >> BGR_TO_YUV_AVERAGING_SHIFT) + Y_ADJUST);
        }

        SIMD_INLINE int BgrToU(int blue, int green, int red)
        {
            return RestrictRange(((BLUE_TO_U_WEIGHT*blue + GREEN_TO_U_WEIGHT * green + RED_TO_U_WEIGHT * red +
                BGR_TO_YUV_ROUND_TERM) >> BGR_TO_YUV_AVERAGING_SHIFT) + UV_ADJUST);
        }

        SIMD_INLINE int BgrToV(int blue, int green, int red)
        {
            return RestrictRange(((BLUE_TO_V_WEIGHT*blue + GREEN_TO_V_WEIGHT * green + RED_TO_V_WEIGHT * red +
                BGR_TO_YUV_ROUND_TERM) >> BGR_TO_YUV_AVERAGING_SHIFT) + UV_ADJUST);
        }

        SIMD_INLINE int YuvToBlue(int y, int u)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + U_TO_BLUE_WEIGHT * (u - UV_ADJUST) +
                YUV_TO_BGR_ROUND_TERM) >> YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int YuvToGreen(int y, int u, int v)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + U_TO_GREEN_WEIGHT * (u - UV_ADJUST) +
                V_TO_GREEN_WEIGHT * (v - UV_ADJUST) + YUV_TO_BGR_ROUND_TERM) >> YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int YuvToRed(int y, int v)
        {
            return RestrictRange((Y_TO_RGB_WEIGHT*(y - Y_ADJUST) + V_TO_RED_WEIGHT * (v - UV_ADJUST) +
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

        SIMD_INLINE void YuvToRgb(int y, int u, int v, uint8_t* rgb)
        {
            rgb[0] = YuvToRed(y, v);
            rgb[1] = YuvToGreen(y, u, v);
            rgb[2] = YuvToBlue(y, u);
        }

        SIMD_INLINE void BgrToHsv(int blue, int green, int red, uint8_t * hsv)
        {
            int max = Max(red, Max(green, blue));
            int min = Min(red, Min(green, blue));
            int range = max - min;

            if (range)
            {
                int dividend;

                if (red == max)
                    dividend = green - blue + 6 * range;
                else if (green == max)
                    dividend = blue - red + 2 * range;
                else
                    dividend = red - green + 4 * range;

                hsv[0] = int(KF_255_DIV_6*dividend / range);
            }
            else
                hsv[0] = 0;

            hsv[1] = max ? 255 * range / max : 0;

            hsv[2] = max;
        }

        SIMD_INLINE void YuvToHsv(int y, int u, int v, uint8_t * hsv)
        {
            int blue = YuvToBlue(y, u);
            int green = YuvToGreen(y, u, v);
            int red = YuvToRed(y, v);
            BgrToHsv(blue, green, red, hsv);
        }

        SIMD_INLINE void BgrToHsl(int blue, int green, int red, uint8_t * hsl)
        {
            int max = Max(red, Max(green, blue));
            int min = Min(red, Min(green, blue));
            int range = max - min;
            int sum = max + min;

            if (range)
            {
                int dividend;

                if (red == max)
                    dividend = green - blue + 6 * range;
                else if (green == max)
                    dividend = blue - red + 2 * range;
                else
                    dividend = red - green + 4 * range;

                hsl[0] = int(KF_255_DIV_6*dividend / range);
            }
            else
                hsl[0] = 0;

            if (sum == 0 || sum == 510)
                hsl[1] = 0;
            else if (sum <= 255)
                hsl[1] = range * 255 / sum;
            else
                hsl[1] = range * 255 / (510 - sum);

            hsl[2] = sum / 2;
        }

        SIMD_INLINE void YuvToHsl(int y, int u, int v, uint8_t * hsl)
        {
            int blue = YuvToBlue(y, u);
            int green = YuvToGreen(y, u, v);
            int red = YuvToRed(y, v);
            BgrToHsl(blue, green, red, hsl);
        }

        SIMD_INLINE void HsvToBgr(int hue, int saturation, int value, uint8_t * bgr)
        {
            if (saturation)
            {
                int sector = hue * 6 / 255;
                int min = (255 - saturation)*value / 255;
                int delta = (value - min)*(hue * 6 - sector * 255) / 255;

                switch (sector)
                {
                case 0:
                    bgr[0] = min;
                    bgr[1] = min + delta;
                    bgr[2] = value;
                    break;
                case 1:
                    bgr[0] = min;
                    bgr[1] = value;
                    bgr[2] = value - delta;
                    break;
                case 2:
                    bgr[0] = min + delta;
                    bgr[1] = value;
                    bgr[2] = min;
                    break;
                case 3:
                    bgr[0] = value;
                    bgr[1] = value - delta;
                    bgr[2] = min;
                    break;
                case 4:
                    bgr[0] = value;
                    bgr[1] = min;
                    bgr[2] = min + delta;
                    break;
                case 5:
                    bgr[0] = value - delta;
                    bgr[1] = min;
                    bgr[2] = value;
                    break;
                default:
                    assert(0);
                }
            }
            else
            {
                bgr[0] = value;
                bgr[1] = value;
                bgr[2] = value;
            }
        }

        SIMD_INLINE void HslToBgr(int hue, int saturation, int lightness, uint8_t * bgr)
        {
            if (saturation)
            {
                int sector = hue * 6 / 255;
                int max;
                if (lightness <= 128)
                    max = lightness * (255 + saturation) / 255;
                else
                    max = ((255 - lightness)*saturation + lightness * 255) / 255;
                int min = (255 - saturation)*max / 255;
                int delta = (max - min)*(hue * 6 - sector * 255) / 255;

                switch (sector)
                {
                case 0:
                    bgr[0] = min;
                    bgr[1] = min + delta;
                    bgr[2] = max;
                    break;
                case 1:
                    bgr[0] = min;
                    bgr[1] = max;
                    bgr[2] = max - delta;
                    break;
                case 2:
                    bgr[0] = min + delta;
                    bgr[1] = max;
                    bgr[2] = min;
                    break;
                case 3:
                    bgr[0] = max;
                    bgr[1] = max - delta;
                    bgr[2] = min;
                    break;
                case 4:
                    bgr[0] = max;
                    bgr[1] = min;
                    bgr[2] = min + delta;
                    break;
                case 5:
                    bgr[0] = max - delta;
                    bgr[1] = min;
                    bgr[2] = max;
                    break;
                default:
                    assert(0);
                }
            }
            else
            {
                bgr[0] = lightness;
                bgr[1] = lightness;
                bgr[2] = lightness;
            }
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

        SIMD_INLINE __m128i BgrToY32(__m128i b16_r16, __m128i g16_1)
        {
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(b16_r16, K16_BY_RY),
                _mm_madd_epi16(g16_1, K16_GY_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgrToY16(__m128i b16, __m128i g16, __m128i r16)
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_Y_ADJUST, _mm_packs_epi32(
                BgrToY32(_mm_unpacklo_epi16(b16, r16), _mm_unpacklo_epi16(g16, K16_0001)),
                BgrToY32(_mm_unpackhi_epi16(b16, r16), _mm_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m128i BgrToY8(__m128i b8, __m128i g8, __m128i r8)
        {
            return _mm_packus_epi16(
                BgrToY16(_mm_unpacklo_epi8(b8, K_ZERO), _mm_unpacklo_epi8(g8, K_ZERO), _mm_unpacklo_epi8(r8, K_ZERO)),
                BgrToY16(_mm_unpackhi_epi8(b8, K_ZERO), _mm_unpackhi_epi8(g8, K_ZERO), _mm_unpackhi_epi8(r8, K_ZERO)));
        }

        SIMD_INLINE __m128i BgrToU32(__m128i b16_r16, __m128i g16_1)
        {
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(b16_r16, K16_BU_RU),
                _mm_madd_epi16(g16_1, K16_GU_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgrToU16(__m128i b16, __m128i g16, __m128i r16)
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(
                BgrToU32(_mm_unpacklo_epi16(b16, r16), _mm_unpacklo_epi16(g16, K16_0001)),
                BgrToU32(_mm_unpackhi_epi16(b16, r16), _mm_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m128i BgrToU8(__m128i b8, __m128i g8, __m128i r8)
        {
            return _mm_packus_epi16(
                BgrToU16(_mm_unpacklo_epi8(b8, K_ZERO), _mm_unpacklo_epi8(g8, K_ZERO), _mm_unpacklo_epi8(r8, K_ZERO)),
                BgrToU16(_mm_unpackhi_epi8(b8, K_ZERO), _mm_unpackhi_epi8(g8, K_ZERO), _mm_unpackhi_epi8(r8, K_ZERO)));
        }

        SIMD_INLINE __m128i BgrToV32(__m128i b16_r16, __m128i g16_1)
        {
            return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(b16_r16, K16_BV_RV),
                _mm_madd_epi16(g16_1, K16_GV_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgrToV16(__m128i b16, __m128i g16, __m128i r16)
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(
                BgrToV32(_mm_unpacklo_epi16(b16, r16), _mm_unpacklo_epi16(g16, K16_0001)),
                BgrToV32(_mm_unpackhi_epi16(b16, r16), _mm_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m128i BgrToV8(__m128i b8, __m128i g8, __m128i r8)
        {
            return _mm_packus_epi16(
                BgrToV16(_mm_unpacklo_epi8(b8, K_ZERO), _mm_unpacklo_epi8(g8, K_ZERO), _mm_unpacklo_epi8(r8, K_ZERO)),
                BgrToV16(_mm_unpackhi_epi8(b8, K_ZERO), _mm_unpackhi_epi8(g8, K_ZERO), _mm_unpackhi_epi8(r8, K_ZERO)));
        }
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
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

        SIMD_INLINE __m128i BgrToBlue(__m128i bgr[3])
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_BLUE),
                    _mm_or_si128(_mm_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_BLUE),
                        _mm_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_BLUE)));
        }

        SIMD_INLINE __m128i BgrToGreen(__m128i bgr[3])
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_GREEN),
                    _mm_or_si128(_mm_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_GREEN),
                        _mm_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_GREEN)));
        }

        SIMD_INLINE __m128i BgrToRed(__m128i bgr[3])
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_RED),
                    _mm_or_si128(_mm_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_RED),
                        _mm_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_RED)));
        }
    }
#endif

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

        SIMD_INLINE __m256i BgrToY32(__m256i b16_r16, __m256i g16_1)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(b16_r16, K16_BY_RY),
                _mm256_madd_epi16(g16_1, K16_GY_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgrToY16(__m256i b16, __m256i g16, __m256i r16)
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_Y_ADJUST, _mm256_packs_epi32(
                BgrToY32(_mm256_unpacklo_epi16(b16, r16), _mm256_unpacklo_epi16(g16, K16_0001)),
                BgrToY32(_mm256_unpackhi_epi16(b16, r16), _mm256_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m256i BgrToY8(__m256i b8, __m256i g8, __m256i r8)
        {
            return _mm256_packus_epi16(
                BgrToY16(_mm256_unpacklo_epi8(b8, K_ZERO), _mm256_unpacklo_epi8(g8, K_ZERO), _mm256_unpacklo_epi8(r8, K_ZERO)),
                BgrToY16(_mm256_unpackhi_epi8(b8, K_ZERO), _mm256_unpackhi_epi8(g8, K_ZERO), _mm256_unpackhi_epi8(r8, K_ZERO)));
        }

        SIMD_INLINE __m256i BgrToU32(__m256i b16_r16, __m256i g16_1)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(b16_r16, K16_BU_RU),
                _mm256_madd_epi16(g16_1, K16_GU_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgrToU16(__m256i b16, __m256i g16, __m256i r16)
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_UV_ADJUST, _mm256_packs_epi32(
                BgrToU32(_mm256_unpacklo_epi16(b16, r16), _mm256_unpacklo_epi16(g16, K16_0001)),
                BgrToU32(_mm256_unpackhi_epi16(b16, r16), _mm256_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m256i BgrToU8(__m256i b8, __m256i g8, __m256i r8)
        {
            return _mm256_packus_epi16(
                BgrToU16(_mm256_unpacklo_epi8(b8, K_ZERO), _mm256_unpacklo_epi8(g8, K_ZERO), _mm256_unpacklo_epi8(r8, K_ZERO)),
                BgrToU16(_mm256_unpackhi_epi8(b8, K_ZERO), _mm256_unpackhi_epi8(g8, K_ZERO), _mm256_unpackhi_epi8(r8, K_ZERO)));
        }

        SIMD_INLINE __m256i BgrToV32(__m256i b16_r16, __m256i g16_1)
        {
            return _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(b16_r16, K16_BV_RV),
                _mm256_madd_epi16(g16_1, K16_GV_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m256i BgrToV16(__m256i b16, __m256i g16, __m256i r16)
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_UV_ADJUST, _mm256_packs_epi32(
                BgrToV32(_mm256_unpacklo_epi16(b16, r16), _mm256_unpacklo_epi16(g16, K16_0001)),
                BgrToV32(_mm256_unpackhi_epi16(b16, r16), _mm256_unpackhi_epi16(g16, K16_0001)))));
        }

        SIMD_INLINE __m256i BgrToV8(__m256i b8, __m256i g8, __m256i r8)
        {
            return _mm256_packus_epi16(
                BgrToV16(_mm256_unpacklo_epi8(b8, K_ZERO), _mm256_unpacklo_epi8(g8, K_ZERO), _mm256_unpacklo_epi8(r8, K_ZERO)),
                BgrToV16(_mm256_unpackhi_epi8(b8, K_ZERO), _mm256_unpackhi_epi8(g8, K_ZERO), _mm256_unpackhi_epi8(r8, K_ZERO)));
        }

        template <int index> __m256i GrayToBgr(__m256i gray);

        template<> SIMD_INLINE __m256i GrayToBgr<0>(__m256i gray)
        {
            return _mm256_shuffle_epi8(_mm256_permute4x64_epi64(gray, 0x44), K8_SHUFFLE_GRAY_TO_BGR0);
        }

        template<> SIMD_INLINE __m256i GrayToBgr<1>(__m256i gray)
        {
            return _mm256_shuffle_epi8(_mm256_permute4x64_epi64(gray, 0x99), K8_SHUFFLE_GRAY_TO_BGR1);
        }

        template<> SIMD_INLINE __m256i GrayToBgr<2>(__m256i gray)
        {
            return _mm256_shuffle_epi8(_mm256_permute4x64_epi64(gray, 0xEE), K8_SHUFFLE_GRAY_TO_BGR2);
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

        SIMD_INLINE __m256i BgrToBlue(__m256i bgr[3])
        {
            __m256i b0 = _mm256_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_BLUE);
            __m256i b2 = _mm256_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_BLUE);
            return
                _mm256_or_si256(_mm256_permute2x128_si256(b0, b2, 0x20),
                    _mm256_or_si256(_mm256_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_BLUE),
                        _mm256_permute2x128_si256(b0, b2, 0x31)));
        }

        SIMD_INLINE __m256i BgrToGreen(__m256i bgr[3])
        {
            __m256i g0 = _mm256_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_GREEN);
            __m256i g2 = _mm256_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_GREEN);
            return
                _mm256_or_si256(_mm256_permute2x128_si256(g0, g2, 0x20),
                    _mm256_or_si256(_mm256_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_GREEN),
                        _mm256_permute2x128_si256(g0, g2, 0x31)));
        }

        SIMD_INLINE __m256i BgrToRed(__m256i bgr[3])
        {
            __m256i r0 = _mm256_shuffle_epi8(bgr[0], K8_SHUFFLE_BGR0_TO_RED);
            __m256i r2 = _mm256_shuffle_epi8(bgr[2], K8_SHUFFLE_BGR2_TO_RED);
            return
                _mm256_or_si256(_mm256_permute2x128_si256(r0, r2, 0x20),
                    _mm256_or_si256(_mm256_shuffle_epi8(bgr[1], K8_SHUFFLE_BGR1_TO_RED),
                        _mm256_permute2x128_si256(r0, r2, 0x31)));
        }

        template<bool tail> __m256i BgrToBgra(const __m256i & bgr, const __m256i & alpha);

        template<> SIMD_INLINE __m256i BgrToBgra<false>(const __m256i & bgr, const __m256i & alpha)
        {
            return _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(bgr, 0x94), K8_BGR_TO_BGRA_SHUFFLE), alpha);
        }

        template<> SIMD_INLINE __m256i BgrToBgra<true>(const __m256i & bgr, const __m256i & alpha)
        {
            return _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(bgr, 0xE9), K8_BGR_TO_BGRA_SHUFFLE), alpha);
        }

        template<bool tail> __m256i RgbToBgra(const __m256i & rgb, const __m256i & alpha);

        template<> SIMD_INLINE __m256i RgbToBgra<false>(const __m256i & rgb, const __m256i & alpha)
        {
            return _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(rgb, 0x94), K8_RGB_TO_BGRA_SHUFFLE), alpha);
        }

        template<> SIMD_INLINE __m256i RgbToBgra<true>(const __m256i & rgb, const __m256i & alpha)
        {
            return _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(rgb, 0xE9), K8_RGB_TO_BGRA_SHUFFLE), alpha);
        }
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i AdjustY16(__m512i y16)
        {
            return _mm512_subs_epi16(y16, K16_Y_ADJUST);
        }

        SIMD_INLINE __m512i AdjustUV16(__m512i uv16)
        {
            return _mm512_subs_epi16(uv16, K16_UV_ADJUST);
        }

        SIMD_INLINE __m512i AdjustedYuvToRed32(__m512i y16_1, __m512i v16_0)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, K16_YRGB_RT),
                _mm512_madd_epi16(v16_0, K16_VR_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m512i AdjustedYuvToRed16(__m512i y16, __m512i v16)
        {
            return SaturateI16ToU8(_mm512_packs_epi32(
                AdjustedYuvToRed32(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(v16, K_ZERO)),
                AdjustedYuvToRed32(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(v16, K_ZERO))));
        }

        SIMD_INLINE __m512i AdjustedYuvToGreen32(__m512i y16_1, __m512i u16_v16)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, K16_YRGB_RT),
                _mm512_madd_epi16(u16_v16, K16_UG_VG)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m512i AdjustedYuvToGreen16(__m512i y16, __m512i u16, __m512i v16)
        {
            return SaturateI16ToU8(_mm512_packs_epi32(
                AdjustedYuvToGreen32(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(u16, v16)),
                AdjustedYuvToGreen32(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(u16, v16))));
        }

        SIMD_INLINE __m512i AdjustedYuvToBlue32(__m512i y16_1, __m512i u16_0)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(y16_1, K16_YRGB_RT),
                _mm512_madd_epi16(u16_0, K16_UB_0)), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m512i AdjustedYuvToBlue16(__m512i y16, __m512i u16)
        {
            return SaturateI16ToU8(_mm512_packs_epi32(
                AdjustedYuvToBlue32(_mm512_unpacklo_epi16(y16, K16_0001), _mm512_unpacklo_epi16(u16, K_ZERO)),
                AdjustedYuvToBlue32(_mm512_unpackhi_epi16(y16, K16_0001), _mm512_unpackhi_epi16(u16, K_ZERO))));
        }

        SIMD_INLINE __m512i YuvToRed(__m512i y, __m512i v)
        {
            __m512i lo = AdjustedYuvToRed16(
                AdjustY16(_mm512_unpacklo_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpacklo_epi8(v, K_ZERO)));
            __m512i hi = AdjustedYuvToRed16(
                AdjustY16(_mm512_unpackhi_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpackhi_epi8(v, K_ZERO)));
            return _mm512_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m512i YuvToGreen(__m512i y, __m512i u, __m512i v)
        {
            __m512i lo = AdjustedYuvToGreen16(
                AdjustY16(_mm512_unpacklo_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpacklo_epi8(u, K_ZERO)),
                AdjustUV16(_mm512_unpacklo_epi8(v, K_ZERO)));
            __m512i hi = AdjustedYuvToGreen16(
                AdjustY16(_mm512_unpackhi_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpackhi_epi8(u, K_ZERO)),
                AdjustUV16(_mm512_unpackhi_epi8(v, K_ZERO)));
            return _mm512_packus_epi16(lo, hi);
        }

        SIMD_INLINE __m512i YuvToBlue(__m512i y, __m512i u)
        {
            __m512i lo = AdjustedYuvToBlue16(
                AdjustY16(_mm512_unpacklo_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpacklo_epi8(u, K_ZERO)));
            __m512i hi = AdjustedYuvToBlue16(
                AdjustY16(_mm512_unpackhi_epi8(y, K_ZERO)),
                AdjustUV16(_mm512_unpackhi_epi8(u, K_ZERO)));
            return _mm512_packus_epi16(lo, hi);
        }

        template <int index> __m512i GrayToBgr(__m512i gray);

        template<> SIMD_INLINE __m512i GrayToBgr<0>(__m512i gray)
        {
            return _mm512_shuffle_epi8(_mm512_shuffle_i64x2(gray, gray, 0x40), K8_SHUFFLE_GRAY_TO_BGR0);
        }

        template<> SIMD_INLINE __m512i GrayToBgr<1>(__m512i gray)
        {
            return _mm512_shuffle_epi8(_mm512_shuffle_i64x2(gray, gray, 0xA5), K8_SHUFFLE_GRAY_TO_BGR1);
        }

        template<> SIMD_INLINE __m512i GrayToBgr<2>(__m512i gray)
        {
            return _mm512_shuffle_epi8(_mm512_shuffle_i64x2(gray, gray, 0xFE), K8_SHUFFLE_GRAY_TO_BGR2);
        }

        SIMD_INLINE __m512i BgrToY32(__m512i b16_r16, __m512i g16_1)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(b16_r16, K16_BY_RY),
                _mm512_madd_epi16(g16_1, K16_GY_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m512i BgrToU32(__m512i b16_r16, __m512i g16_1)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(b16_r16, K16_BU_RU),
                _mm512_madd_epi16(g16_1, K16_GU_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m512i BgrToV32(__m512i b16_r16, __m512i g16_1)
        {
            return _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(b16_r16, K16_BV_RV),
                _mm512_madd_epi16(g16_1, K16_GV_RT)), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        template <int index> __m512i InterleaveBgr(__m512i blue, __m512i green, __m512i red);

        template<> SIMD_INLINE __m512i InterleaveBgr<0>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, blue), K8_SHUFFLE_BLUE_TO_BGR0),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, green), K8_SHUFFLE_GREEN_TO_BGR0),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, red), K8_SHUFFLE_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m512i InterleaveBgr<1>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, blue), K8_SHUFFLE_BLUE_TO_BGR1),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, green), K8_SHUFFLE_GREEN_TO_BGR1),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, red), K8_SHUFFLE_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m512i InterleaveBgr<2>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, blue), K8_SHUFFLE_BLUE_TO_BGR2),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, green), K8_SHUFFLE_GREEN_TO_BGR2),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, red), K8_SHUFFLE_RED_TO_BGR2)));
        }
    }
#endif//SIMD_AVX512BW_ENABLE 

#ifdef SIMD_VMX_ENABLE    
    namespace Vmx
    {
        SIMD_INLINE v128_s16 AdjustY(v128_u16 y)
        {
            return vec_sub((v128_s16)y, K16_Y_ADJUST);
        }

        SIMD_INLINE v128_s16 AdjustUV(v128_u16 uv)
        {
            return vec_sub((v128_s16)uv, K16_UV_ADJUST);
        }

        SIMD_INLINE v128_s32 PreparedYuvToRed(v128_s16 y_1, v128_s16 v_0)
        {
            return vec_sra(vec_msum(y_1, K16_YRGB_RT, vec_msum(v_0, K16_VR_0, (v128_s32)K32_00000000)), K32_YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 AdjustedYuvToRed(v128_s16 y, v128_s16 v)
        {
            v128_s32 lo = PreparedYuvToRed((v128_s16)UnpackLoU16(K16_0001, (v128_u16)y), (v128_s16)UnpackLoU16(K16_0000, (v128_u16)v));
            v128_s32 hi = PreparedYuvToRed((v128_s16)UnpackHiU16(K16_0001, (v128_u16)y), (v128_s16)UnpackHiU16(K16_0000, (v128_u16)v));
            return SaturateI16ToU8(vec_pack(lo, hi));
        }

        SIMD_INLINE v128_u8 YuvToRed(v128_u8 y, v128_u8 v)
        {
            v128_u16 lo = AdjustedYuvToRed(AdjustY(UnpackLoU8(y)), AdjustUV(UnpackLoU8(v)));
            v128_u16 hi = AdjustedYuvToRed(AdjustY(UnpackHiU8(y)), AdjustUV(UnpackHiU8(v)));
            return vec_pack(lo, hi);
        }

        SIMD_INLINE v128_s32 PreparedYuvToGreen(v128_s16 y_1, v128_s16 u_v)
        {
            return vec_sra(vec_msum(y_1, K16_YRGB_RT, vec_msum(u_v, K16_UG_VG, (v128_s32)K32_00000000)), K32_YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 AdjustedYuvToGreen(v128_s16 y, v128_s16 u, v128_s16 v)
        {
            v128_s32 lo = PreparedYuvToGreen((v128_s16)UnpackLoU16(K16_0001, (v128_u16)y), (v128_s16)UnpackLoU16((v128_u16)v, (v128_u16)u));
            v128_s32 hi = PreparedYuvToGreen((v128_s16)UnpackHiU16(K16_0001, (v128_u16)y), (v128_s16)UnpackHiU16((v128_u16)v, (v128_u16)u));
            return SaturateI16ToU8(vec_pack(lo, hi));
        }

        SIMD_INLINE v128_u8 YuvToGreen(v128_u8 y, v128_u8 u, v128_u8 v)
        {
            v128_u16 lo = AdjustedYuvToGreen(AdjustY(UnpackLoU8(y)), AdjustUV(UnpackLoU8(u)), AdjustUV(UnpackLoU8(v)));
            v128_u16 hi = AdjustedYuvToGreen(AdjustY(UnpackHiU8(y)), AdjustUV(UnpackHiU8(u)), AdjustUV(UnpackHiU8(v)));
            return vec_pack(lo, hi);
        }

        SIMD_INLINE v128_s32 PreparedYuvToBlue(v128_s16 y_1, v128_s16 u_0)
        {
            return vec_sra(vec_msum(y_1, K16_YRGB_RT, vec_msum(u_0, K16_UB_0, (v128_s32)K32_00000000)), K32_YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 AdjustedYuvToBlue(v128_s16 y, v128_s16 u)
        {
            v128_s32 lo = PreparedYuvToBlue((v128_s16)UnpackLoU16(K16_0001, (v128_u16)y), (v128_s16)UnpackLoU16(K16_0000, (v128_u16)u));
            v128_s32 hi = PreparedYuvToBlue((v128_s16)UnpackHiU16(K16_0001, (v128_u16)y), (v128_s16)UnpackHiU16(K16_0000, (v128_u16)u));
            return SaturateI16ToU8(vec_pack(lo, hi));
        }

        SIMD_INLINE v128_u8 YuvToBlue(v128_u8 y, v128_u8 u)
        {
            v128_u16 lo = AdjustedYuvToBlue(AdjustY(UnpackLoU8(y)), AdjustUV(UnpackLoU8(u)));
            v128_u16 hi = AdjustedYuvToBlue(AdjustY(UnpackHiU8(y)), AdjustUV(UnpackHiU8(u)));
            return vec_pack(lo, hi);
        }

        SIMD_INLINE v128_s32 BgrToY(v128_s16 b_r, v128_s16 g_1)
        {
            return vec_sra(vec_msum(b_r, K16_BY_RY, vec_msum(g_1, K16_GY_RT, (v128_s32)K32_00000000)), K32_BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 BgrToY(v128_s16 b, v128_s16 g, v128_s16 r)
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_Y_ADJUST, vec_pack(
                BgrToY((v128_s16)UnpackLoU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackLoU16(K16_0001, (v128_u16)g)),
                BgrToY((v128_s16)UnpackHiU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackHiU16(K16_0001, (v128_u16)g)))));
        }

        SIMD_INLINE v128_u8 BgrToY(v128_u8 b, v128_u8 g, v128_u8 r)
        {
            return vec_pack(
                BgrToY((v128_s16)UnpackLoU8(b), (v128_s16)UnpackLoU8(g), (v128_s16)UnpackLoU8(r)),
                BgrToY((v128_s16)UnpackHiU8(b), (v128_s16)UnpackHiU8(g), (v128_s16)UnpackHiU8(r)));
        }

        SIMD_INLINE v128_s32 BgrToU(v128_s16 b_r, v128_s16 g_1)
        {
            return vec_sra(vec_msum(b_r, K16_BU_RU, vec_msum(g_1, K16_GU_RT, (v128_s32)K32_00000000)), K32_BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 BgrToU(v128_s16 b, v128_s16 g, v128_s16 r)
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_UV_ADJUST, vec_pack(
                BgrToU((v128_s16)UnpackLoU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackLoU16(K16_0001, (v128_u16)g)),
                BgrToU((v128_s16)UnpackHiU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackHiU16(K16_0001, (v128_u16)g)))));
        }

        SIMD_INLINE v128_u8 BgrToU(v128_u8 b, v128_u8 g, v128_u8 r)
        {
            return vec_pack(
                BgrToU((v128_s16)UnpackLoU8(b), (v128_s16)UnpackLoU8(g), (v128_s16)UnpackLoU8(r)),
                BgrToU((v128_s16)UnpackHiU8(b), (v128_s16)UnpackHiU8(g), (v128_s16)UnpackHiU8(r)));
        }

        SIMD_INLINE v128_s32 BgrToV(v128_s16 b_r, v128_s16 g_1)
        {
            return vec_sra(vec_msum(b_r, K16_BV_RV, vec_msum(g_1, K16_GV_RT, (v128_s32)K32_00000000)), K32_BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE v128_u16 BgrToV(v128_s16 b, v128_s16 g, v128_s16 r)
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_UV_ADJUST, vec_pack(
                BgrToV((v128_s16)UnpackLoU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackLoU16(K16_0001, (v128_u16)g)),
                BgrToV((v128_s16)UnpackHiU16((v128_u16)r, (v128_u16)b), (v128_s16)UnpackHiU16(K16_0001, (v128_u16)g)))));
        }

        SIMD_INLINE v128_u8 BgrToV(v128_u8 b, v128_u8 g, v128_u8 r)
        {
            return vec_pack(
                BgrToV((v128_s16)UnpackLoU8(b), (v128_s16)UnpackLoU8(g), (v128_s16)UnpackLoU8(r)),
                BgrToV((v128_s16)UnpackHiU8(b), (v128_s16)UnpackHiU8(g), (v128_s16)UnpackHiU8(r)));
        }

        template <int index> v128_u8 InterleaveBgr(v128_u8 blue, v128_u8 green, v128_u8 red);

        template<> SIMD_INLINE v128_u8 InterleaveBgr<0>(v128_u8 blue, v128_u8 green, v128_u8 red)
        {
            return vec_perm(vec_perm(blue, green, K8_PERM_INTERLEAVE_BGR_00), red, K8_PERM_INTERLEAVE_BGR_01);
        }

        template<> SIMD_INLINE v128_u8 InterleaveBgr<1>(v128_u8 blue, v128_u8 green, v128_u8 red)
        {
            return vec_perm(vec_perm(blue, green, K8_PERM_INTERLEAVE_BGR_10), red, K8_PERM_INTERLEAVE_BGR_11);
        }

        template<> SIMD_INLINE v128_u8 InterleaveBgr<2>(v128_u8 blue, v128_u8 green, v128_u8 red)
        {
            return vec_perm(vec_perm(blue, green, K8_PERM_INTERLEAVE_BGR_20), red, K8_PERM_INTERLEAVE_BGR_21);
        }

        SIMD_INLINE v128_u8 BgrToBlue(v128_u8 bgr[3])
        {
            return vec_perm(vec_perm(bgr[0], bgr[1], K8_PERM_BGR_TO_BLUE_0), bgr[2], K8_PERM_BGR_TO_BLUE_1);
        }

        SIMD_INLINE v128_u8 BgrToGreen(v128_u8 bgr[3])
        {
            return vec_perm(vec_perm(bgr[0], bgr[1], K8_PERM_BGR_TO_GREEN_0), bgr[2], K8_PERM_BGR_TO_GREEN_1);
        }

        SIMD_INLINE v128_u8 BgrToRed(v128_u8 bgr[3])
        {
            return vec_perm(vec_perm(bgr[0], bgr[1], K8_PERM_BGR_TO_RED_0), bgr[2], K8_PERM_BGR_TO_RED_1);
        }
    }
#endif// SIMD_VMX_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <int part> SIMD_INLINE uint32x4_t BgrToGray(const uint16x8_t & blue, const uint16x8_t & green, const uint16x8_t & red)
        {
            return vshrq_n_u32(vmlal_u16(vmlal_u16(vmlal_u16(K32_BGR_TO_GRAY_ROUND_TERM, Half<part>(blue), K16_BLUE_TO_GRAY_WEIGHT),
                Half<part>(green), K16_GREEN_TO_GRAY_WEIGHT), Half<part>(red), K16_RED_TO_GRAY_WEIGHT), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE uint16x8_t BgrToGray(const uint16x8_t & blue, const uint16x8_t & green, const uint16x8_t & red)
        {
            return PackU32(BgrToGray<0>(blue, green, red), BgrToGray<1>(blue, green, red));
        }

        template <int part> SIMD_INLINE int32x4_t BgrToY(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(vmlal_s16(K32_BGR_TO_YUV_ROUND_TERM, vreinterpret_s16_u16(Half<part>(blue)), K16_BLUE_TO_Y_WEIGHT),
                vreinterpret_s16_u16(Half<part>(green)), K16_GREEN_TO_Y_WEIGHT), vreinterpret_s16_u16(Half<part>(red)), K16_RED_TO_Y_WEIGHT), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t BgrToY(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vaddq_s16(K16_Y_ADJUST, PackI32(BgrToY<0>(blue, green, red), BgrToY<1>(blue, green, red)));
        }

        SIMD_INLINE uint8x16_t BgrToY(uint8x16_t blue, uint8x16_t green, uint8x16_t red)
        {
            return PackSaturatedI16(
                BgrToY(UnpackU8<0>(blue), UnpackU8<0>(green), UnpackU8<0>(red)),
                BgrToY(UnpackU8<1>(blue), UnpackU8<1>(green), UnpackU8<1>(red)));
        }

        template <int part> SIMD_INLINE int32x4_t BgrToU(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(vmlal_s16(K32_BGR_TO_YUV_ROUND_TERM, vreinterpret_s16_u16(Half<part>(blue)), K16_BLUE_TO_U_WEIGHT),
                vreinterpret_s16_u16(Half<part>(green)), K16_GREEN_TO_U_WEIGHT), vreinterpret_s16_u16(Half<part>(red)), K16_RED_TO_U_WEIGHT), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t BgrToU(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vaddq_s16(K16_UV_ADJUST, PackI32(BgrToU<0>(blue, green, red), BgrToU<1>(blue, green, red)));
        }

        SIMD_INLINE uint8x16_t BgrToU(uint8x16_t blue, uint8x16_t green, uint8x16_t red)
        {
            return PackSaturatedI16(
                BgrToU(UnpackU8<0>(blue), UnpackU8<0>(green), UnpackU8<0>(red)),
                BgrToU(UnpackU8<1>(blue), UnpackU8<1>(green), UnpackU8<1>(red)));
        }

        template <int part> SIMD_INLINE int32x4_t BgrToV(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(vmlal_s16(K32_BGR_TO_YUV_ROUND_TERM, vreinterpret_s16_u16(Half<part>(blue)), K16_BLUE_TO_V_WEIGHT),
                vreinterpret_s16_u16(Half<part>(green)), K16_GREEN_TO_V_WEIGHT), vreinterpret_s16_u16(Half<part>(red)), K16_RED_TO_V_WEIGHT), Base::BGR_TO_YUV_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t BgrToV(uint16x8_t blue, uint16x8_t green, uint16x8_t red)
        {
            return vaddq_s16(K16_UV_ADJUST, PackI32(BgrToV<0>(blue, green, red), BgrToV<1>(blue, green, red)));
        }

        SIMD_INLINE uint8x16_t BgrToV(uint8x16_t blue, uint8x16_t green, uint8x16_t red)
        {
            return PackSaturatedI16(
                BgrToV(UnpackU8<0>(blue), UnpackU8<0>(green), UnpackU8<0>(red)),
                BgrToV(UnpackU8<1>(blue), UnpackU8<1>(green), UnpackU8<1>(red)));
        }

        template <int part> SIMD_INLINE int16x8_t AdjustY(uint8x16_t y)
        {
            return vsubq_s16(vreinterpretq_s16_u16(UnpackU8<part>(y)), K16_Y_ADJUST);
        }

        template <int part> SIMD_INLINE int16x8_t AdjustUV(uint8x16_t uv)
        {
            return vsubq_s16(vreinterpretq_s16_u16(UnpackU8<part>(uv)), K16_UV_ADJUST);
        }

        template <int part> SIMD_INLINE int32x4_t YuvToRed(int16x8_t y, int16x8_t v)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(K32_YUV_TO_BGR_ROUND_TERM, Half<part>(y), K16_Y_TO_RGB_WEIGHT),
                Half<part>(v), K16_V_TO_RED_WEIGHT), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t YuvToRed(int16x8_t y, int16x8_t v)
        {
            return PackI32(YuvToRed<0>(y, v), YuvToRed<1>(y, v));
        }

        SIMD_INLINE uint8x16_t YuvToRed(uint8x16_t y, uint8x16_t v)
        {
            return PackSaturatedI16(YuvToRed(AdjustY<0>(y), AdjustUV<0>(v)), YuvToRed(AdjustY<1>(y), AdjustUV<1>(v)));
        }

        template <int part> SIMD_INLINE int32x4_t YuvToGreen(int16x8_t y, int16x8_t u, int16x8_t v)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(vmlal_s16(K32_YUV_TO_BGR_ROUND_TERM, Half<part>(y), K16_Y_TO_RGB_WEIGHT),
                Half<part>(u), K16_U_TO_GREEN_WEIGHT), Half<part>(v), K16_V_TO_GREEN_WEIGHT), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t YuvToGreen(int16x8_t y, int16x8_t u, int16x8_t v)
        {
            return PackI32(YuvToGreen<0>(y, u, v), YuvToGreen<1>(y, u, v));
        }

        SIMD_INLINE uint8x16_t YuvToGreen(uint8x16_t y, uint8x16_t u, uint8x16_t v)
        {
            return PackSaturatedI16(YuvToGreen(AdjustY<0>(y), AdjustUV<0>(u), AdjustUV<0>(v)),
                YuvToGreen(AdjustY<1>(y), AdjustUV<1>(u), AdjustUV<1>(v)));
        }

        template <int part> SIMD_INLINE int32x4_t YuvToBlue(int16x8_t y, int16x8_t u)
        {
            return vshrq_n_s32(vmlal_s16(vmlal_s16(K32_YUV_TO_BGR_ROUND_TERM, Half<part>(y), K16_Y_TO_RGB_WEIGHT),
                Half<part>(u), K16_U_TO_BLUE_WEIGHT), Base::YUV_TO_BGR_AVERAGING_SHIFT);
        }

        SIMD_INLINE int16x8_t YuvToBlue(int16x8_t y, int16x8_t u)
        {
            return PackI32(YuvToBlue<0>(y, u), YuvToBlue<1>(y, u));
        }

        SIMD_INLINE uint8x16_t YuvToBlue(uint8x16_t y, uint8x16_t u)
        {
            return PackSaturatedI16(YuvToBlue(AdjustY<0>(y), AdjustUV<0>(u)), YuvToBlue(AdjustY<1>(y), AdjustUV<1>(u)));
        }

        SIMD_INLINE void YuvToBgr(uint8x16_t y, uint8x16_t u, uint8x16_t v, uint8x16x3_t & bgr)
        {
            int16x8_t yLo = AdjustY<0>(y), uLo = AdjustUV<0>(u), vLo = AdjustUV<0>(v);
            int16x8_t yHi = AdjustY<1>(y), uHi = AdjustUV<1>(u), vHi = AdjustUV<1>(v);
            bgr.val[0] = PackSaturatedI16(YuvToBlue(yLo, uLo), YuvToBlue(yHi, uHi));
            bgr.val[1] = PackSaturatedI16(YuvToGreen(yLo, uLo, vLo), YuvToGreen(yHi, uHi, vHi));
            bgr.val[2] = PackSaturatedI16(YuvToRed(yLo, vLo), YuvToRed(yHi, vHi));
        }

        SIMD_INLINE void YuvToRgb(uint8x16_t y, uint8x16_t u, uint8x16_t v, uint8x16x3_t& rgb)
        {
            int16x8_t yLo = AdjustY<0>(y), uLo = AdjustUV<0>(u), vLo = AdjustUV<0>(v);
            int16x8_t yHi = AdjustY<1>(y), uHi = AdjustUV<1>(u), vHi = AdjustUV<1>(v);
            rgb.val[0] = PackSaturatedI16(YuvToRed(yLo, vLo), YuvToRed(yHi, vHi));
            rgb.val[1] = PackSaturatedI16(YuvToGreen(yLo, uLo, vLo), YuvToGreen(yHi, uHi, vHi));
            rgb.val[2] = PackSaturatedI16(YuvToBlue(yLo, uLo), YuvToBlue(yHi, uHi));
        }
    }
#endif// SIMD_NEON_ENABLE
}
#endif//__SimdConversion_h__
