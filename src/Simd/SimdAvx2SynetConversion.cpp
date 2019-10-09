/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void StoreScaled(float * ptr, __m256i value32, __m256 scale, __m256 shift)
        {
            Avx::Store<align>(ptr, _mm256_fmadd_ps(_mm256_cvtepi32_ps(value32), scale, shift));
        }

        const __m256i K16_BLUE_RED = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const __m256i K16_GREEN_0000 = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m256i K32_ROUND_TERM = SIMD_MM256_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m256i BgraToGray32(__m256i bgra)
        {
            const __m256i g0a0 = _mm256_and_si256(_mm256_srli_si256(bgra, 1), K16_00FF);
            const __m256i b0r0 = _mm256_and_si256(bgra, K16_00FF);
            const __m256i weightedSum = _mm256_add_epi32(_mm256_madd_epi16(g0a0, K16_GREEN_0000), _mm256_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm256_srli_epi32(_mm256_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInput1(const uint8_t * src, __m256 scale, __m256 shift, float * dst);

        SIMD_INLINE void SynetSetInput1Gray8(__m128i gray8, __m256 scale, __m256 shift, float * dst)
        {
            StoreScaled<false>(dst + 0, _mm256_cvtepu8_epi32(_mm_srli_si128(gray8, 0)), scale, shift);
            StoreScaled<false>(dst + F, _mm256_cvtepu8_epi32(_mm_srli_si128(gray8, 8)), scale, shift);
        }

        SIMD_INLINE void SynetSetInput1Gray8(__m256i gray8, __m256 scale, __m256 shift, float * dst)
        {
            SynetSetInput1Gray8(_mm256_extractf128_si256(gray8, 0), scale, shift, dst + 0 * F);
            SynetSetInput1Gray8(_mm256_extractf128_si256(gray8, 1), scale, shift, dst + 2 * F);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatGray8>(const uint8_t * src, __m256 scale, __m256 shift, float * dst)
        {
            SynetSetInput1Gray8(Sse2::Load<false>((__m128i*)src + 0), scale, shift, dst + 0 * F);
            SynetSetInput1Gray8(Sse2::Load<false>((__m128i*)src + 1), scale, shift, dst + 2 * F);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgr24>(const uint8_t * src, __m256 scale, __m256 shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, BgraToGray32(BgrToBgra<false>(Load<false>((__m256i*)(src + 0)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(BgrToBgra<false>(Load<false>((__m256i*)(src + 24)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(BgrToBgra<false>(Load<false>((__m256i*)(src + 48)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(BgrToBgra<true>(Load<false>((__m256i*)(src + 64)), K32_01000000)), scale, shift);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgra32>(const uint8_t * src, __m256 scale, __m256 shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, BgraToGray32(Load<false>((__m256i*)src + 0)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(Load<false>((__m256i*)src + 1)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(Load<false>((__m256i*)src + 2)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(Load<false>((__m256i*)src + 3)), scale, shift);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatRgb24>(const uint8_t * src, __m256 scale, __m256 shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, BgraToGray32(RgbToBgra<false>(Load<false>((__m256i*)(src + 0)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgraToGray32(RgbToBgra<false>(Load<false>((__m256i*)(src + 24)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 2 * F, BgraToGray32(RgbToBgra<false>(Load<false>((__m256i*)(src + 48)), K32_01000000)), scale, shift);
            StoreScaled<false>(dst + 3 * F, BgraToGray32(RgbToBgra<true>(Load<false>((__m256i*)(src + 64)), K32_01000000)), scale, shift);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInput1(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            __m256 _scale = _mm256_set1_ps(scale[0]);
            __m256 _shift = _mm256_set1_ps(shift[0]);
            size_t aligned = AlignLo(width, A);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInput1<format>(src + step * x, _scale, _shift, dst + x);
                if(aligned < width)
                    SynetSetInput1<format>(src + step * (width - A), _scale, _shift, dst + width - A);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNchw3(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel);

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatGray8>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel)
        {
            __m128i src0 = Sse2::Load<false>((__m128i*)src + 0);
            __m256i gray0 = _mm256_cvtepu8_epi32(_mm_srli_si128(src0, 0));
            __m256i gray1 = _mm256_cvtepu8_epi32(_mm_srli_si128(src0, 8));
            __m128i src1 = Sse2::Load<false>((__m128i*)src + 1);
            __m256i gray2 = _mm256_cvtepu8_epi32(_mm_srli_si128(src1, 0));
            __m256i gray3 = _mm256_cvtepu8_epi32(_mm_srli_si128(src1, 8));
            StoreScaled<false>(dst + 0 * F, gray0, scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[0], shift[0]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[0], shift[0]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[0], shift[0]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[1], shift[1]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[1], shift[1]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[1], shift[1]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[2], shift[2]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[2], shift[2]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[2], shift[2]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatBgr24>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel)
        {
            __m256i _bgr[3];
            _bgr[0] = Load<false>((__m256i*)src + 0);
            _bgr[1] = Load<false>((__m256i*)src + 1);
            _bgr[2] = Load<false>((__m256i*)src + 2);
            SynetSetInput1Gray8(BgrToBlue(_bgr), scale[0], shift[0], dst + 0 * channel);
            SynetSetInput1Gray8(BgrToGreen(_bgr), scale[1], shift[1], dst + 1 * channel);
            SynetSetInput1Gray8(BgrToRed(_bgr), scale[2], shift[2], dst + 2 * channel);
        }

        SIMD_INLINE void SynetSetInputNchw3Bgra32(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel)
        {
            __m256i bgra = Load<false>((__m256i*)src);
            StoreScaled<false>(dst + 0 * channel, _mm256_and_si256(_mm256_srli_si256(bgra, 0), K32_000000FF), scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * channel, _mm256_and_si256(_mm256_srli_si256(bgra, 1), K32_000000FF), scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * channel, _mm256_and_si256(_mm256_srli_si256(bgra, 2), K32_000000FF), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatBgra32>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel)
        {
            SynetSetInputNchw3Bgra32(src + 0 * A, scale, shift, dst + 0 * F, channel);
            SynetSetInputNchw3Bgra32(src + 1 * A, scale, shift, dst + 1 * F, channel);
            SynetSetInputNchw3Bgra32(src + 2 * A, scale, shift, dst + 2 * F, channel);
            SynetSetInputNchw3Bgra32(src + 3 * A, scale, shift, dst + 3 * F, channel);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatRgb24>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst, size_t channel)
        {
            __m256i _rgb[3];
            _rgb[0] = Load<false>((__m256i*)src + 0);
            _rgb[1] = Load<false>((__m256i*)src + 1);
            _rgb[2] = Load<false>((__m256i*)src + 2);
            SynetSetInput1Gray8(BgrToRed(_rgb), scale[0], shift[0], dst + 0 * channel);
            SynetSetInput1Gray8(BgrToGreen(_rgb), scale[1], shift[1], dst + 1 * channel);
            SynetSetInput1Gray8(BgrToBlue(_rgb), scale[2], shift[2], dst + 2 * channel);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t aligned = AlignLo(width, A), channel = width * height;
            __m256 _scale[3], _shift[3];
            for (size_t i = 0; i < 3; ++i)
            {
                _scale[i] = _mm256_set1_ps(scale[i]);
                _shift[i] = _mm256_set1_ps(shift[i]);
            }
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInputNchw3<format>(src + step * x, _scale, _shift, dst + x, channel);
                if (aligned < width)
                    SynetSetInputNchw3<format>(src + step * (width - A), _scale, _shift, dst + width - A, channel);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNhwc3(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatGray8>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst)
        {
            __m128i gray0 = Sse2::Load<false>((__m128i*)src + 0);
            __m128i bgr0 = _mm_shuffle_epi8(gray0, Ssse3::K8_SHUFFLE_GRAY_TO_BGR0);
            StoreScaled<false>(dst + 0x0 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr0, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr0, 8)), scale[1], shift[1]);
            __m128i bgr1 = _mm_shuffle_epi8(gray0, Ssse3::K8_SHUFFLE_GRAY_TO_BGR1);
            StoreScaled<false>(dst + 0x2 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr1, 0)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr1, 8)), scale[0], shift[0]);
            __m128i bgr2 = _mm_shuffle_epi8(gray0, Ssse3::K8_SHUFFLE_GRAY_TO_BGR2);
            StoreScaled<false>(dst + 0x4 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr2, 0)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr2, 8)), scale[2], shift[2]);
            __m128i gray1 = Sse2::Load<false>((__m128i*)src + 1);
            __m128i bgr3 = _mm_shuffle_epi8(gray1, Ssse3::K8_SHUFFLE_GRAY_TO_BGR0);
            StoreScaled<false>(dst + 0x6 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr3, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr3, 8)), scale[1], shift[1]);
            __m128i bgr4 = _mm_shuffle_epi8(gray1, Ssse3::K8_SHUFFLE_GRAY_TO_BGR1);
            StoreScaled<false>(dst + 0x8 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr4, 0)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr4, 8)), scale[0], shift[0]);
            __m128i bgr5 = _mm_shuffle_epi8(gray1, Ssse3::K8_SHUFFLE_GRAY_TO_BGR2);
            StoreScaled<false>(dst + 0xA * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr5, 0)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr5, 8)), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatBgr24>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst)
        {
            __m128i bgr0 = Sse2::Load<false>((__m128i*)src + 0);
            StoreScaled<false>(dst + 0x0 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr0, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr0, 8)), scale[1], shift[1]);
            __m128i bgr1 = Sse2::Load<false>((__m128i*)src + 1);
            StoreScaled<false>(dst + 0x2 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr1, 0)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr1, 8)), scale[0], shift[0]);
            __m128i bgr2 = Sse2::Load<false>((__m128i*)src + 2);
            StoreScaled<false>(dst + 0x4 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr2, 0)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr2, 8)), scale[2], shift[2]);
            __m128i bgr3 = Sse2::Load<false>((__m128i*)src + 3);
            StoreScaled<false>(dst + 0x6 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr3, 0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr3, 8)), scale[1], shift[1]);
            __m128i bgr4 = Sse2::Load<false>((__m128i*)src + 4);
            StoreScaled<false>(dst + 0x8 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr4, 0)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr4, 8)), scale[0], shift[0]);
            __m128i bgr5 = Sse2::Load<false>((__m128i*)src + 5);
            StoreScaled<false>(dst + 0xA * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr5, 0)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm256_cvtepu8_epi32(_mm_srli_si128(bgr5, 8)), scale[2], shift[2]);
        }

        const __m128i K8_BGRA_TO_BGR_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_BGRA_TO_BGR_1 = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x3, 0x4, 0x6, 0x7, 0x8, 0xA, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_BGRA_TO_BGR_2 = SIMD_MM_SETR_EPI8(0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1, -1, -1, -1, -1);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatBgra32>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst)
        {
            StoreScaled<false>(dst + 0x0 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 0)), K8_BGRA_TO_BGR_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 10)), K8_BGRA_TO_BGR_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 16)), K8_BGRA_TO_BGR_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 32)), K8_BGRA_TO_BGR_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x4 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 42)), K8_BGRA_TO_BGR_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 48)), K8_BGRA_TO_BGR_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x6 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 64)), K8_BGRA_TO_BGR_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 74)), K8_BGRA_TO_BGR_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x8 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 80)), K8_BGRA_TO_BGR_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 96)), K8_BGRA_TO_BGR_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 106)), K8_BGRA_TO_BGR_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm256_cvtepu8_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 112)), K8_BGRA_TO_BGR_2)), scale[2], shift[2]);
        }

        const __m128i K8_RGB_UNPACK_0 = SIMD_MM_SETR_EPI8(0x2, -1, 0x1, -1, 0x0, -1, 0x5, -1, 0x4, - 1, 0x3, -1, 0x8, -1, 0x7, -1);
        const __m128i K8_RGB_UNPACK_1 = SIMD_MM_SETR_EPI8(0x0, -1, 0x5, -1, 0x4, -1, 0x3, -1, 0x8, - 1, 0x7, -1, 0x6, -1, 0xB, -1);
        const __m128i K8_RGB_UNPACK_2 = SIMD_MM_SETR_EPI8(0x8, -1, 0x7, -1, 0xC, -1, 0xB, -1, 0xA, - 1, 0xF, -1, 0xE, -1, 0xD, -1);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatRgb24>(const uint8_t * src, const __m256 * scale, const __m256 * shift, float * dst)
        {
            StoreScaled<false>(dst + 0x0 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 0)), K8_RGB_UNPACK_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 6)), K8_RGB_UNPACK_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x2 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 8)), K8_RGB_UNPACK_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 24)), K8_RGB_UNPACK_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x4 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 30)), K8_RGB_UNPACK_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 32)), K8_RGB_UNPACK_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x6 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 48)), K8_RGB_UNPACK_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 54)), K8_RGB_UNPACK_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x8 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 56)), K8_RGB_UNPACK_2)), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 72)), K8_RGB_UNPACK_0)), scale[0], shift[0]);
            StoreScaled<false>(dst + 0xA * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 78)), K8_RGB_UNPACK_1)), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, _mm256_cvtepi16_epi32(_mm_shuffle_epi8(Sse2::Load<false>((__m128i*)(src + 80)), K8_RGB_UNPACK_2)), scale[2], shift[2]);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t aligned = AlignLo(width, A);
            __m256 _scale[3], _shift[3];
            _scale[0] = _mm256_setr_ps(scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1]);
            _scale[1] = _mm256_setr_ps(scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0]);
            _scale[2] = _mm256_setr_ps(scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2]);
            _shift[0] = _mm256_setr_ps(shift[0], shift[1], shift[2], shift[0], shift[1], shift[2], shift[0], shift[1]);
            _shift[1] = _mm256_setr_ps(shift[2], shift[0], shift[1], shift[2], shift[0], shift[1], shift[2], shift[0]);
            _shift[2] = _mm256_setr_ps(shift[1], shift[2], shift[0], shift[1], shift[2], shift[0], shift[1], shift[2]);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInputNhwc3<format>(src + step * x, _scale, _shift, dst + 3 * x);
                if (aligned < width)
                    SynetSetInputNhwc3<format>(src + step * (width - A), _scale, _shift, dst + 3 * (width - A));
                src += stride;
                dst += 3*width;
            }
        }

        void SynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            assert(width >= A);

            float scale[3];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SynetSetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SynetSetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SynetSetInput1<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SynetSetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
            default: assert(0);
            }
        }
    }
#endif//SIMD_AVX2_ENABLE
}
