/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <class T> SIMD_INLINE __m256i UnpackY(__m256i uyvy)
        {
            static const __m256i Y_SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1,
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1);
            static const __m256i Y_LO = SIMD_MM256_SET1_EPI16(T::Y_LO);
            return _mm256_subs_epi16(_mm256_shuffle_epi8(uyvy, Y_SHUFFLE), Y_LO);
        }

        template <class T> SIMD_INLINE __m256i UnpackU(__m256i uyvy)
        {
            static const __m256i U_SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1,
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1);
            static const __m256i U_Z = SIMD_MM256_SET1_EPI16(T::UV_Z);
            return _mm256_subs_epi16(_mm256_shuffle_epi8(uyvy, U_SHUFFLE), U_Z);
        }

        template <class T> SIMD_INLINE __m256i UnpackV(__m256i uyvy)
        {
            static const __m256i V_SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1,
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1);
            static const __m256i V_Z= SIMD_MM256_SET1_EPI16(T::UV_Z);
            return _mm256_subs_epi16(_mm256_shuffle_epi8(uyvy, V_SHUFFLE), V_Z);
        }

        template <bool align, class T> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr)
        {
            __m256i uyvy0 = Load<align>((__m256i*)uyvy + 0);
            __m256i y0 = UnpackY<T>(uyvy0);
            __m256i u0 = UnpackU<T>(uyvy0);
            __m256i v0 = UnpackV<T>(uyvy0);
            __m256i blue0 = YuvToBlue16<T>(y0, u0);
            __m256i green0 = YuvToGreen16<T>(y0, u0, v0);
            __m256i red0 = YuvToRed16<T>(y0, v0);

            __m256i uyvy1 = Load<align>((__m256i*)uyvy + 1);
            __m256i y1 = UnpackY<T>(uyvy1);
            __m256i u1 = UnpackU<T>(uyvy1);
            __m256i v1 = UnpackV<T>(uyvy1);
            __m256i blue1 = YuvToBlue16<T>(y1, u1);
            __m256i green1 = YuvToGreen16<T>(y1, u1, v1);
            __m256i red1 = YuvToRed16<T>(y1, v1);

            __m256i blue = _mm256_permute4x64_epi64(_mm256_packus_epi16(blue0, blue1), 0xD8);
            __m256i green = _mm256_permute4x64_epi64(_mm256_packus_epi16(green0, green1), 0xD8);
            __m256i red = _mm256_permute4x64_epi64(_mm256_packus_epi16(red0, red1), 0xD8);
            Store<align>((__m256i*)bgr + 0, InterleaveBgr<0>(blue, green, red));
            Store<align>((__m256i*)bgr + 1, InterleaveBgr<1>(blue, green, red));
            Store<align>((__m256i*)bgr + 2, InterleaveBgr<2>(blue, green, red));
        }

        template <bool align, class T> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2 * A));
            if (align)
                assert(Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t sizeS = width * 2, sizeD = width * 3;
            size_t sizeS2A = AlignLo(sizeS, 2 * A);
            size_t tailS = sizeS - 2 * A;
            size_t tailD = sizeD - 3 * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colS = 0, colD = 0; colS < sizeS2A; colS += 2 * A, colD += 3 * A)
                    Uyvy422ToBgr<align, T>(uyvy + colS, bgr + colD);
                if (sizeS2A != sizeS)
                    Uyvy422ToBgr<false, T>(uyvy + tailS, bgr + tailD);
                uyvy += uyvyStride;
                bgr += bgrStride;
            }
        }

        template<bool align> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Uyvy422ToBgr<align, Base::Bt601>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Uyvy422ToBgr<align, Base::Bt709>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Uyvy422ToBgr<align, Base::Bt2020>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Uyvy422ToBgr<align, Base::Trect871>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride))
                Uyvy422ToBgr<true>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
            else
                Uyvy422ToBgr<false>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
        }
    }
#endif
}
